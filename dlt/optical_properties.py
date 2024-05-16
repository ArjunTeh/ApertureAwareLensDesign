import jax
import jax.numpy as jnp

from . import zemax_loader
from . import aspheric
from . import curvature_sphere
from . import sources
from . import zemax
from . import tracing
from . import constants


def load_state(lens_desc, wavelength):
    return zemax_loader.desc_state_to_array(lens_desc, wavelength=wavelength, config_id=0)


def trace_lens(lens_desc, wavelength=587.6, reverse=False, beam_size=1.0):
    lens = load_state(lens_desc, wavelength=wavelength)[0][:, :4]
    return trace_state(lens, wavelength=wavelength, reverse=reverse, beam_size=beam_size)


def trace_state(lens, wavelength=587.6, reverse=False, beam_size=1.0, asphere=False):
    raygen = sources.line_source_quadrature(beam_size * lens[:, 1].max(), 0.0)
    x, v, L, *ray_aux = raygen(20, key=jax.random.PRNGKey(0))

    surfs = zemax.zemax_state_to_jax_surfaces(lens)
    if asphere:
        surf_funs = [aspheric.functionSuite('standard')] * len(surfs)
    else:
        surf_funs = [curvature_sphere.functionSuite()] * len(surfs)

    if reverse:
        surf_funs = surf_funs[::-1]
        surfs = surfs[::-1]
        x = x.at[:, 0].set(lens[:, 1].sum() + 5.0)
        v = -v

    vtracer = jax.vmap(tracing.trace_with_warpfield, (None, None, 0, 0, 0))
    valid, xp, vp, Lp, Tx, wargmin = vtracer(surf_funs, surfs, x, v, L)

    return valid, xp, vp, Lp


def effective_focal_length(lens_desc, wavelength=587.6):
    ivalid, img_xp, img_vp, img_Lp = trace_lens(lens_desc, beam_size=0.1, wavelength=wavelength, reverse=True)
    ovalid, obj_xp, obj_vp, obj_Lp = trace_lens(lens_desc, beam_size=0.1, wavelength=wavelength, reverse=False)

    img_plane_samples = (obj_xp[0][ovalid, 1] / obj_vp[-1][ovalid, 1]) * (-obj_vp[-1][ovalid, 0])
    obj_plane_samples = (img_xp[0][ivalid, 1] / img_vp[-1][ivalid, 1]) * (img_vp[-1][ivalid, 0])


    effective_back_focal_length = img_plane_samples.mean()
    effective_front_focal_length = obj_plane_samples.mean()
    return effective_front_focal_length, effective_back_focal_length


def effective_focal_length_from_state(lens_state, wavelength=587.6, asphere=False):
    ivalid, img_xp, img_vp, img_Lp = trace_state(lens_state, beam_size=0.1, wavelength=wavelength, reverse=True, asphere=asphere)
    ovalid, obj_xp, obj_vp, obj_Lp = trace_state(lens_state, beam_size=0.1, wavelength=wavelength, reverse=False, asphere=asphere)

    img_plane_samples = (obj_xp[0][:, 1] / obj_vp[-1][:, 1]) * (-obj_vp[-1][:, 0])
    obj_plane_samples = (img_xp[0][:, 1] / img_vp[-1][:, 1]) * (img_vp[-1][:, 0])

    effective_back_focal_length = jnp.sum(jnp.where(ovalid, img_plane_samples, 0.0)) / jnp.maximum(jnp.count_nonzero(ovalid), 1.0)
    effective_front_focal_length = jnp.sum(jnp.where(ivalid, obj_plane_samples, 0.0)) / jnp.maximum(jnp.count_nonzero(ivalid), 1.0)
    return effective_front_focal_length, effective_back_focal_length


def effective_focal_length_zoom_state(lens_state, surf_ids, distances, wavelength=587.6, asphere=False):
    efls = []
    ebls = []
    for d in distances:
        lens_state = lens_state.at[surf_ids, 1].set(d)
        efl, ebl = effective_focal_length_from_state(lens_state, wavelength=wavelength, asphere=asphere)
        efls.append(efl)
        ebls.append(ebl)
    return efls, ebls


def effective_focal_length_magnification_from_state(lens_state, wavelength=587.6):
    # m = x / f where m is magnification, x is extension from image plane, f is focal length

    od = 1000
    extension = (od * 50) / (od - 50) - 50
    # extension = 2.63 # about focus for 1 meter away
    lens_modded = jnp.array(lens_state)
    lens_modded = lens_modded.at[-1, 1].add(extension)
    surfs = zemax.zemax_state_to_jax_surfaces(lens_modded)

    surf_funs = [curvature_sphere.functionSuite()] * len(surfs)

    sample_height = 0.01
    ray_samples = 10
    sample_heights = jnp.linspace(0, sample_height, ray_samples+1)[1:]
    x = jnp.zeros((ray_samples, 3)).at[:, 0].set(-5.0)
    x = x.at[:, 1].set(sample_heights)
    v = jnp.zeros_like(x).at[:, 0].set(1.0)
    L = jnp.ones((ray_samples,))

    vtracer = jax.vmap(tracing.trace_scan_no_stop, (None, None, 0, 0, 0, None))
    (valid, xp, vp, Lp), (Tx, wargmin) = vtracer(surf_funs[0], surfs, x, v, L, 0)
    magnification = -xp[:, -1, 1] / sample_heights

    efl = extension / magnification
    return efl.mean()


def move_sensor_plane(lens_state, wavelength=587.56):
    if lens_state.shape[1] > 4:
        raise ValueError("lens_state should only have 4 columns, asphere not supported yet")

    surfs = zemax.zemax_state_to_jax_surfaces(lens_state)
    surf_funs = curvature_sphere.functionSuite() # this means only curvature sphere for now

    sample_height = 0.5*lens_state[0, 3]
    ray_samples = 100
    sample_heights = jnp.linspace(0, sample_height, ray_samples+1)[1:]
    x = jnp.zeros((ray_samples, 3)).at[:, 0].set(-5.0)
    x = x.at[:, 1].set(sample_heights)
    v = jnp.zeros_like(x).at[:, 0].set(1.0)
    L = jnp.ones((ray_samples,))

    vtracer = jax.vmap(tracing.trace_scan_no_stop, (None, None, 0, 0, 0, None))
    (valid, xp, vp, Lp), (Tx, wargmin) = vtracer(surf_funs, surfs, x, v, L, 0)

    # take the last directions and positions and intersect the optical axis
    weights = 1 / sample_heights[valid]
    end_dir = vp[valid, -2, :]
    end_pos = xp[valid, -2, :]

    end_pos_intersect = end_pos - (end_pos[:, 1:2] / end_dir[:, 1:2]) * end_dir
    sensor_point = end_pos_intersect[:, 0]
    sensor_point = jnp.sum(sensor_point * weights) / jnp.sum(weights)

    right_vertex = lens_state[:-1, 1].sum()

    new_lens = lens_state.copy()
    new_lens = new_lens.at[-1, 1].set(sensor_point - right_vertex)

    return new_lens


def effective_semidiameter(k1, k2, th):
    ''' calculate the largest diameter that this lens can have'''
    no_intersect = jnp.logical_and(k1 < 0, k2 > 0)

    square_term = -th*(k1*th - 2)*(k2*th + 2)*(k1*k2*th + 2*k1 - 2*k2)
    denom = k1*k2*th + k1 - k2

    invalid = jnp.logical_or(square_term < 0, jnp.isclose(denom, 0))

    square_term = jnp.where(invalid, constants.FLOAT_BIG_NUMBER, square_term)
    denom = jnp.where(invalid, 1e-6, denom)

    max_height = (1/2) * jnp.sqrt(square_term) / denom
    return jnp.abs(max_height)


def max_curvature_for_semidiameter(k1, k2, th, semidiam):
    ap = semidiam

    hsquare_term = -th*(k1*th - 2)*(k2*th + 2)*(k1*k2*th + 2*k1 - 2*k2)
    hdenom = k1*k2*th + k1 - k2
    invalid = jnp.logical_or(hsquare_term < 0, jnp.isclose(hdenom, 0))

    ap_square_term = jnp.maximum(1 - (ap*k2)**2, 0.0)
    invalid = jnp.logical_or(invalid, (ap_square_term < 0))

    # expression generated symbolically
    num = 2*(th*jnp.sqrt(ap_square_term)*(k2*th + 2) + (k2*th + 1)*(2*ap**2*k2 + k2*th**2 + 2*th))
    den = (4*ap**2*k2**2*th**2 + 8*ap**2*k2*th + 4*ap**2 + k2**2*th**4 + 4*k2*th**3 + 4*th**2)

    k1soln = jnp.where(invalid, k1, num / den)
    return k1soln


def optimize_sensor_plane(loss_and_grad_fn, lens, eps=0.01):
    if not isinstance(lens, jnp.ndarray):
        lens = jnp.array(lens)

    def change_dist_loss(t, key):
        lens_sens = lens.at[-1, 1].set(t)
        (val, aux), grad = loss_and_grad_fn(lens_sens, 0.0, key)
        grad = grad[-1, 1]
        return val, grad

    sens_dist = optimize(change_dist_loss, lens[-1, 1], eps)

    return lens.at[-1, 1].set(sens_dist)


def optimize(loss_and_grad_fn, t_init, eps, niters=200, key_init=jax.random.PRNGKey(0)):
    tdelt = 0
    key = key_init
    for i in range(niters):
        key, subkey = jax.random.split(key)
        loss, grad = loss_and_grad_fn(t_init + tdelt, subkey)
        tdelt = tdelt - eps * grad / jnp.sqrt(i + 1)

    best_dist = t_init + tdelt
    return best_dist
