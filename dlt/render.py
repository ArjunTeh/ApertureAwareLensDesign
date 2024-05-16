import jax
import jax.numpy as jnp
import jax.scipy as jsp
import cv2
import tqdm

from functools import partial
from typing import NamedTuple

from . import curvature_sphere
from . import tracing
from . import sensor
from . import sources
from . import optical_properties
from . import zemax


class QuantizedRect(NamedTuple):
    height: float
    hres: int
    width: float
    wres: int
    distance: float
    cone_angle: float


def _get_points(src : QuantizedRect):
    Xm, Ym = jnp.meshgrid(jnp.arange(src.hres), jnp.arange(src.wres))
    Xf = ((Xm.flatten() + 0.5) / src.hres - 0.5) * src.height
    Yf = ((Ym.flatten() + 0.5) / src.wres - 0.5) * src.width
    # Xm, Ym = jnp.meshgrid(jnp.linspace(-src.height/2, src.height/2, src.hres), jnp.linspace(-src.width/2, src.width/2, src.wres))
    # return Xm.flatten(), Ym.flatten()
    return Xf, Yf


def image_sampler(fname, src_height, src_width, color=False):
    # read it in black and white?
    # create function the properly interpolates
    if fname is not None:
        if color:
            image = cv2.imread(fname)
        else:
            image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        imres = image.shape[:2]
    else:
        image = jnp.ones((10, 10))
        imres = (10, 10)

    image = image.astype(jnp.float32) / 255.0

    sampler_order = 1
    def sample_fun_color(h, w):
        ucoord = (h + src_height/2) / src_height * imres[0]
        vcoord = (w + src_width/2) / src_width * imres[1]
        map_coords_color = jax.vmap(partial(jsp.ndimage.map_coordinates, order=sampler_order, mode='constant', cval=0), in_axes=(-1, None))
        Lvals = map_coords_color(image, jnp.stack([ucoord, vcoord]))
        return Lvals

    def sample_fun(h, w):
        ucoord = (h + src_height/2) / src_height * imres[0] - 0.5
        vcoord = (w + src_width/2) / src_width * imres[1] - 0.5
        Lvals = jsp.ndimage.map_coordinates(image, jnp.stack([ucoord, vcoord]), order=sampler_order, mode='constant', cval=0)
        return Lvals

    return sample_fun_color if color else sample_fun


def sample_plane_points(rng_key, surf_to_sample : QuantizedRect, spp):
    rngkey, subkey = jax.random.split(rng_key)
    total_samples = surf_to_sample.hres * surf_to_sample.wres * spp 
    
    pixel_size = jnp.array([surf_to_sample.height / surf_to_sample.hres, surf_to_sample.width / surf_to_sample.wres])

    X, Y = _get_points(surf_to_sample)

    sample_perturbation = jax.random.uniform(subkey, (total_samples, 2), minval=-0.5, maxval=0.5) * pixel_size[None, :]

    surf_pos = jnp.stack([
        jnp.ones((total_samples)) * surf_to_sample.distance,
        jnp.tile(X, spp) + sample_perturbation[:, 0],
        jnp.tile(Y, spp) + sample_perturbation[:, 1],
    ], axis=-1)

    return surf_pos


def point_src_to_lens(rng_key, pixel_pos, surf_to_sample : QuantizedRect, spp, pixel_size):
    rngkey, subkey1, subkey2 = jax.random.split(rng_key, 3)
    total_samples = spp
    surf_area = surf_to_sample.height * surf_to_sample.width
    surf_importance = 1 / surf_area
    sampled_points = jax.random.uniform(subkey1, (total_samples, 2), minval=-0.5, maxval=0.5)
    sampled_points = sampled_points * jnp.array([surf_to_sample.height, surf_to_sample.width])

    sample_perturb = jax.random.uniform(subkey2, (total_samples, 2), minval=-0.5, maxval=0.5) * jnp.array(pixel_size)
    pixel_importance =  1 / (pixel_size[0] * pixel_size[1])

    surf_pos = jnp.stack([
        jnp.ones((sampled_points.shape[0])) * surf_to_sample.distance,
        sampled_points[:, 0],
        sampled_points[:, 1],
    ], axis=-1)

    pixel_perturb_pos = jnp.tile(pixel_pos, (total_samples, 1))
    pixel_perturb_pos = pixel_perturb_pos.at[:, 1:].add(sample_perturb)
    vel_to_surf = (surf_pos - pixel_perturb_pos)
    dist = jnp.linalg.norm(vel_to_surf, axis=-1)
    vel_to_surf = vel_to_surf / dist[:, None]

    # start_pos = jnp.tile(pixel_pos, (total_samples, 1))
    start_pos = pixel_perturb_pos

    importance = (jnp.abs(vel_to_surf[:, 0]) / dist**2) * surf_importance * pixel_importance
    L = jnp.ones_like(importance)
    return start_pos, vel_to_surf, L, importance


def render_zemax_image(lens, src : QuantizedRect, sens : QuantizedRect, image_sampler, spp=100, rng_key=None):

    effl, ebfl = optical_properties.effective_focal_length_from_state(lens)
    surfs = zemax.zemax_state_to_jax_surfaces(lens, scene_scale=1.0, focal_length=effl, object_distance=src.distance)
    surf_funs = curvature_sphere.functionSuite()

    srcX, srcY = _get_points(src)
    srcL = image_sampler(srcX, srcY)

    nrays = spp
    vrender_point = jax.vmap(render_point, (0, 0, 0, None, None, None, None, None, None, None))
    image = vrender_point(srcX, srcY, srcL, nrays, src.distance, src.cone_angle, surf_funs, surfs, rng_key, True)

    # image = jnp.ones((sens.hres, sens.wres))
    # sensor.sensor_splat(image, imsize=(sens.height, sens.width), bottom_left=(surfs[-1][0], -5, -5), ray=(xp[:, -1, :], vp[:, -1, :]), L=image)
    return image, (srcX, srcY)


def render_zemax_image_farfield(lens, sens : QuantizedRect, image_sampler, rng_key, spp=100, nbatches=1, verbose=False, nowarp=False) -> jnp.ndarray:
    # setup tracer for lens
    surfs = zemax.zemax_state_to_jax_surfaces(lens, scene_scale=1.0)
    surfs = surfs[-2::-1] # ignore the sensor surface since we are tracing from there
    surf_funs = curvature_sphere.functionSuite()
    tracer_fun = partial(trace_rays_to_sensor, surf_funs, surfs, stop_id=None)
    tracer_fun = jax.jit(tracer_fun)
    pixel_size = [(sens.height / sens.hres), (sens.width / sens.wres)]

    # setup tracing
    target_surf = QuantizedRect(2.5 * lens[-1, 3], 100, 2.5 * lens[-1, 3], 100, lens[:-1, 1].sum(), jnp.pi/8)
    vsrc = jax.vmap(partial(point_src_to_lens, spp=spp, surf_to_sample=target_surf, pixel_size=pixel_size), in_axes=(0, 0))

    # setup sensor image splatter
    splatter = partial(sensor.sensor_splat, imsize=(sens.height, sens.width), bottom_left=(target_surf.distance, -sens.height/2, -sens.width/2))
    splatter = jax.jit(splatter, donate_argnums=0)

    image = jnp.zeros((sens.hres, sens.wres))
    ray_start = lens[:, 1].sum()
    srcX, srcY = _get_points(sens)
    src_positions = jnp.stack([jnp.ones_like(srcX)*ray_start, srcX, srcY], axis=-1)
    for positions in tqdm.tqdm(jnp.split(src_positions, nbatches, axis=0), disable=not verbose):

        rng_keys = jax.random.split(rng_key, positions.shape[0]+1)
        rng_key, subkeys = rng_keys[0], rng_keys[1:]
        x, v, L, importance = vsrc(subkeys, positions)
        x = x.reshape((-1, 3))
        v = v.reshape((-1, 3))
        L = L.reshape((-1,))
        importance = importance.reshape((-1,))
        rays = (x, v, L)

        valid, xt, vt, Lt, detTx = tracer_fun(rays)
        valid_ratio = jnp.count_nonzero(valid) / valid.shape[0]

        unit_hemi = 2 * jnp.pi
        if nowarp:
            detTx = 1.0
        endL = unit_hemi * detTx * importance * Lt * image_sampler(vt[:, 1], vt[:, 2]) * jnp.abs(v[:, 0])
        image = splatter(image, pos=x, L=endL)

    return image / spp, valid_ratio


def render_zemax_throughput(lens, src, sens, spp=100, resolution=100, rng_key=None):
    src_height, src_width, src_distance, cone_angle = src

    effl, ebfl = optical_properties.effective_focal_length_from_state(lens)
    surfs = zemax.zemax_state_to_jax_surfaces(lens, scene_scale=1.0, focal_length=effl, object_distance=src_distance)
    surf_funs = curvature_sphere.functionSuite()

    res = 100
    Xm, Ym = jnp.meshgrid(jnp.linspace(-src_height/2, src_height/2, res), jnp.linspace(-src_width/2, src_width/2, res))
    Lvals = jnp.ones_like(Xm.flatten())

    nrays = spp
    keys = jax.random.split(rng_key, Lvals.shape[0])
    vrender_point = jax.vmap(render_point, (0, 0, 0, None, None, None, None, None, 0, None, None))
    image = vrender_point(Xm.flatten(), Ym.flatten(), Lvals, nrays, src_distance, cone_angle, surf_funs, surfs, keys, False, False)
    return image, Xm.flatten(), Ym.flatten()


def render_zemax_spoterror(lens, src, sens, spp=100, resolution=100, rng_key=None):
    src_height, src_width, src_distance, cone_angle = src

    effl, ebfl = optical_properties.effective_focal_length_from_state(lens)
    surfs = zemax.zemax_state_to_jax_surfaces(lens, scene_scale=1.0, focal_length=effl, object_distance=src_distance)
    surf_funs = curvature_sphere.functionSuite()

    res = 100
    Xm, Ym = jnp.meshgrid(jnp.linspace(-src_height/2, src_height/2, res), jnp.linspace(-src_width/2, src_width/2, res))
    Lvals = jnp.ones_like(Xm.flatten())

    nrays = spp
    keys = jax.random.split(rng_key, Lvals.shape[0])
    vrender_point = jax.vmap(render_point, (0, 0, 0, None, None, None, None, None, 0, None, None))
    image = vrender_point(Xm.flatten(), Ym.flatten(), Lvals, nrays, src_distance, cone_angle, surf_funs, surfs, keys, False, True)
    return image, Xm.flatten(), Ym.flatten()


def render_point(h, w, L, nrays, src_dist, cone_angle, surf_funs, surfs, rng_key, include_radiance, spot_error):
    first_semi = surfs[0][1]
    src_pos = jnp.array([-src_dist, h, w])
    src_dir = -src_pos / jnp.linalg.norm(src_pos)
    
    # pick a direction such that is covers the lens

    up = jnp.array([0, 1, 0])
    (x, v, Luni)= sources.square_point_random(nrays=nrays, position=src_pos, direction=src_dir, up=up, cone_angle=cone_angle, rng_key=rng_key)
    rays = (x, v, L*Luni)
    if spot_error:
        return render_point_blur(surf_funs, surfs, rays, stop_id=None, include_radiance=include_radiance)
    else:
        return render_surf_point(surf_funs, surfs, rays, stop_id=None, include_radiance=include_radiance)


def render_surf_point(surf_funs, surfs, rays, stop_id=None, include_radiance=False):
    
    valid, xt, vt, L, detTx = trace_rays_to_sensor(surf_funs, surfs, rays, stop_id=stop_id)

    endL = L if include_radiance else jnp.ones_like(L)
    throughput = jnp.sum(endL * detTx, where=valid) / xt.shape[0]
    return throughput


def render_point_blur(surf_funs, surfs, rays, stop_id=None, include_radiance=False):

    valid, xt, vt, L, detTx = trace_rays_to_sensor(surf_funs, surfs, rays, stop_id=stop_id)

    xcenter = jnp.average(xt[:, 1:], axis=0, keepdims=True, weights=valid)
    spot_error = jnp.sum(detTx[:, None]*(xt[:, 1:] - xcenter)**2, where=valid[:, None]) / xt.shape[0]
    return spot_error


def trace_rays_to_sensor(surf_funs, surfs, rays, stop_id=None):
    if stop_id is None:
        vtracer = jax.vmap(tracing.trace_scan_no_stop, (None, None, 0, 0, 0, None))
    else:
        vtracer = jax.vmap(tracing.trace_scan_with_stop_check, (None, None, 0, 0, 0, None))
    
    x, v, L = rays
    (valid, xp, vp, Lp), (Tx, wargmin) = vtracer(surf_funs, surfs, x, v, L, stop_id)

    Lend = jnp.where(valid, Lp[:, -1], 0)
    detTx = jnp.where(valid, jnp.linalg.det(Tx), 0)

    return valid, xp[:, -1, :], vp[:, -1, :], Lend, detTx

