import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from dlt import curvature_sphere
from dlt import optimize
from dlt import sources
from dlt import tracing
from dlt import zemax
from dlt import zemax_loader


def target_positions(nangles, target_focal_length):
    target_heights = [target_focal_length * jnp.tan(ang) for ang in nangles]
    return target_heights


def create_source(line_range, focal_length, object_distance=jnp.inf, angle_list=[0.0], source_type='line', stochastic=False, key=jax.random.PRNGKey(0)):
    angles = angle_list
    if stochastic:
        line_generators = [sources.line_source_stratified_random(line_range, a) for a in angles]
    else:
        if source_type == 'line':
            if object_distance == jnp.inf:
                line_generators = [sources.line_source_quadrature(line_range, a) for a in angles]
            else:
                line_generators = [sources.point_source_quadrature(line_range, a, object_distance) for a in angles]
        elif source_type == 'leggauss':
            line_generators = [sources.leggaus_circular_quadrature(line_range, 6, 4, a) for a in angles]
        else:
            raise ValueError(f"source type not recognized: {source_type}")

    def generate_rays(nrays, key):
        rays = [lg(nrays//len(angles), key) for lg in line_generators]
        x = jnp.concatenate([r[0] for r in rays], axis=0)
        v = jnp.concatenate([r[1] for r in rays], axis=0)
        L = jnp.concatenate([r[2] for r in rays], axis=0)
        w = jnp.concatenate([r[3] for r in rays], axis=0)
        rad = jnp.concatenate([r[4] for r in rays], axis=0)

        target_heights = target_positions(angles, target_focal_length=focal_length)
        targets = jnp.zeros(x.shape)
        for i, th in enumerate(target_heights):
            step = x.shape[0] // len(angles)
            targets = targets.at[i*step:(i+1)*step, 1].set(th)
            # targets[i*step:(i+1)*step, 1] = target_heights[i]

        return x, v, L, jnp.asarray(targets), w, jnp.abs(rad), line_range

    return generate_rays


def create_source_from_params(lens, **params):
    focal_length = params.get('focal_length', 20.0)
    angle_list = params.get('angle_list', [0.0])
    source_type = params.get('source_type', 'line')
    beam_size = params.get('beam_size', 1.0)
    line_range = lens[:, 3].max() * beam_size
    object_distance = params.get('object_distance', np.inf)
    stochastic = params.get('stochastic', False)
    ray_generator = create_source(line_range, 
                                  angle_list=angle_list, 
                                  focal_length=focal_length, 
                                  object_distance=object_distance, 
                                  source_type=source_type, 
                                  stochastic=stochastic)
    return ray_generator


def create_tracing_loss(lens : np.ndarray, blur_loss_fn, **params):
    nrays = params.get('nrays', 100)
    loss_fn = create_tracing_loss_rays(lens, blur_loss_fn, **params)

    generate_rays = create_source_from_params(lens, **params)

    def loss(state, na_exp, rng_key):
        key, subkey = jax.random.split(rng_key)
        rays = generate_rays(nrays, key=subkey)
        return loss_fn(state, na_exp, rays)

    return loss

def create_tracing_loss_rays(lens : np.ndarray, blur_loss_fn, **params):
    loss_type = params.get('loss_type', 'blur')
    focal_length = params.get('focal_length', 20.0)
    angle_list = params.get('angle_list', [0.0])
    object_distance = params.get('object_distance', np.inf)
    surface_type_list = params.get('surface_type_list', None)
    nrays = params.get('nrays', 100)

    surf_funs = curvature_sphere.functionSuite()
    stop_idx = surface_type_list.index(zemax_loader.ZemaxSurfaceType.STOP)
    vtracer = jax.vmap(tracing.trace_scan_with_stop_check, (None, None, 0, 0, 0, None))

    def loss(state, na_exp, rays):
        surfs = zemax.zemax_state_to_jax_surfaces(state, scene_scale=1.0, focal_length=focal_length, object_distance=object_distance, reparameterize=True)

        x, v, L, targets, weights, radius, line_range = rays
        (valid, xp, vp, Lp), (Tx, wargmin) = vtracer(surf_funs, surfs, x, v, L, stop_idx)
        xp = xp.swapaxes(0, 1)
        Lp = Lp.swapaxes(0, 1)
        vp = vp.swapaxes(0, 1)

        xp_valid = jnp.where(valid[:, None], xp[-1], 0.0)
        targets = jnp.where(valid[:, None], targets, 0.0)

        valid = jax.lax.stop_gradient(valid)
        radius = jnp.where(valid, radius, 0)

        # differentiating the determinant of the jacobian of the warp field yields the trace of the warpfield jacobian
        # which is equivalent to taking the divergence of the warp field directly.
        change_of_meas = jnp.where(valid, jnp.linalg.det(Tx), 0)
        weights_sum = weights.sum()

        throughput_integrand = line_range*2*np.pi*weights*radius*change_of_meas / weights_sum
        throughput = jnp.sum(throughput_integrand)

        spot_centers = None
        spot_errors = None
        spot_error = None

        if loss_type == 'spot_error':
            rays_per_angle = jnp.split(xp_valid, len(angle_list), axis=0)
            valid_per_angle = jnp.split(valid, len(angle_list), axis=0)
            spot_centers = [jnp.average(xview, axis=0, weights=validview)
                            for xview, validview in zip(rays_per_angle, valid_per_angle)]
            spot_errors = [jnp.sum(jnp.where(validview[:, None], (sc - xview)**2, 0.0))
                           for sc, xview, validview in zip(spot_centers, rays_per_angle, valid_per_angle)]
            spot_error = sum(spot_errors) / len(angle_list)
            final_loss = sum(spot_errors) / len(angle_list) - na_exp*throughput
        elif loss_type == 'spot_error_unbiased':
            spot_errors = []
            spot_centers = []
            for ang in range(len(angle_list)):
                idx_slice = slice(ang*nrays//len(angle_list), (ang+1)*nrays//len(angle_list))
                ray_view = xp_valid[idx_slice, :]
                spot_thruput = jnp.where(jnp.any(valid[idx_slice]), jnp.sum(radius[idx_slice] * change_of_meas[idx_slice]), 1.0)
                spot_center = jnp.sum(ray_view * (radius[idx_slice] * change_of_meas[idx_slice])[:, None], axis=0, keepdims=True) / jnp.where(jnp.any(valid[idx_slice]), spot_thruput, 1.0)
                spot_error_ray = 2 * np.pi * radius[idx_slice] * change_of_meas[idx_slice] *((ray_view[:, 1:] - spot_center[:, 1:])**2).sum(axis=-1)
                spot_error = jnp.sum(jnp.where(valid[idx_slice], spot_error_ray, 0.0)) / nrays
                spot_errors.append(spot_error)
                spot_centers.append(spot_center)
            final_loss = sum(spot_errors) - na_exp*throughput

        aux_dict = dict(
            throughput=throughput,
            spot_centers=spot_centers,
            spot_error=spot_error,
            spot_errors=spot_errors,
            valid=valid
        )

        return final_loss, aux_dict

    return loss


def box_aperture_constraints(state, min_bounds, max_bounds):
    astate = state.copy()
    k = state[:, 0]
    ap = state[:, 3]
    out_mask = (k * ap) > 1
    new_k = jnp.where(jnp.signbit(k), -1 / ap, 1 / ap)
    new_k = jnp.where(out_mask, new_k, k)
    astate = astate.at[:, 0].set(new_k)
    return jnp.clip(astate, a_min=min_bounds, a_max=max_bounds)


@jax.jit
def constraint_function(lens):
    min_bounds = -10000 * np.ones(lens.shape[1])
    min_bounds[:4] = [-1.0, 0.01, 1.0, 0.1]
    min_bounds[6:] = -1e-3
    min_bounds = np.tile(min_bounds, (lens.shape[0], 1))

    max_bounds = 10000 * np.ones(lens.shape[1])
    max_bounds[:4] = [1.0, 100.0, 2.0, 100.0]
    max_bounds[6:] = 1e-3
    max_bounds = np.tile(max_bounds, (lens.shape[0], 1))

    return box_aperture_constraints(lens, min_bounds, max_bounds)


def grad_project(state, tangent, surf_list):
    newtan = tangent.at[:, 2].set(0.0) # no change to ior

    for i, s in enumerate(surf_list):
        if s == zemax_loader.ZemaxSurfaceType.STOP:
            newtan = newtan.at[i, 0].set(0.0)
    
    return newtan


def run_opt(lens : np.ndarray, loss_fn, lens_desc, niters=1000, lr=1e-4):
    lens_jnp = jnp.asarray(lens)
    grad_project_fn = jax.jit(lambda s, g : grad_project(s, g, lens_desc))
    output = optimize.run_adam_descent(loss_fn, 
                                    lens,
                                    niters,
                                    constraint=constraint_function,
                                    projector=grad_project_fn,
                                    eps=lr, 
                                    show_progress=True,
                                    save_cadence=5)


    # output is 
    # loss_hist, state_hist, other_info_hist, grad_hist, mo1, mo2
    return output


def get_spot_data(lens, nrings, nangs, max_rad):
    surfs = zemax.zemax_state_to_jax_surfaces(lens)
    surf_funs = curvature_sphere.functionSuite()
    raygen = sources.leggaus_circular_quadrature(max_rad, nrings, nangs, src_angle=np.deg2rad(0.0))
    x, v, L, weights, x_rad = raygen(0, None)
    # trace_fun = jax.vmap(tracing.trace_by_scan, in_axes=(None, None, 0, 0, 0))
    trace_fun = jax.vmap(tracing.trace_scan_no_stop, in_axes=(None, None, 0, 0, 0, None))
    trace_fun_jit = jax.jit(lambda x, v, L: trace_fun(surf_funs, surfs, x, v, L, 0.0))
    (valid, xp, vp, Lp), warpfield = trace_fun_jit(x, v, L)
    xf = xp[valid, -1, 1:]
    return xf[:, 0], xf[:, 1], x_rad[valid]


def plot_spot_diagrams(lenses, names, nrings, nangs, max_rad):

    trace_data = [get_spot_data(lens, nrings, nangs, max_rad) for lens in lenses]

    max_x_val = max([x.max() for x, y, r in trace_data])

    for (x, y, r), name in zip(trace_data, names):
        plt.figure()
        plt.scatter(x, y, c=r, vmin=0.0, vmax=max_rad)
        plt.colorbar()
        plt.xlim(-max_x_val, max_x_val)
        plt.ylim(-max_x_val, max_x_val)
        plt.savefig(f"{name}.png")
        plt.close()


if __name__ == '__main__':
    jax.config.update("jax_debug_nans", True)

    # the experiments require double precision for numerical stability
    jax.config.update("jax_enable_x64", True)

    # opt parameters
    rng_key_init = jax.random.PRNGKey(0)
    niters = 1000
    na_weights = [0.0, 0.2, 0.5]
    na_weights_normalized = [nw / 1000.0 for nw in na_weights]
    lr = 1e-3
    angle_list_degrees = [0.0, 10.0, 20.0]
    angle_list = [np.deg2rad(a) for a in angle_list_degrees]
    nrays = 200 * len(angle_list)

    lens_fname = 'data/zemax_files/double-gauss-50mm.zmx'
    lens_init, lens_desc = zemax_loader.load_zemax_file(lens_fname, info_list=True)
    lens_desc_generic = zemax_loader.load_zemax_generic(lens_fname)

    lens_orig = lens_init.copy()
    max_aperture = lens_init[0, 3]


    loss_params_uniform = dict(
        loss_type='spot_error_unbiased',
        source_type='line',
        nrays=nrays,
        angle_list=angle_list,
        asphere=False,
        object_distance = np.inf,
        stochastic=True,
        log_loss=False,
        surface_type_list=lens_desc,
    )

    def l2_loss_fn(x):
        return jnp.sum(x[:, 1:]**2, axis=-1)

    def list_of_dict_transpose(list_of_dict):
        return {k: np.stack([d[k] for d in list_of_dict]) for k in list_of_dict[0].keys()}

    # run optimization for different na weights
    trace_loss = create_tracing_loss(lens_init, l2_loss_fn, **loss_params_uniform)
    trace_loss_and_grad = jax.jit(jax.value_and_grad(trace_loss, has_aux=True))
    def run_opt_with_na_weight(lens_start, na_weight):
        return run_opt(lens_start, lambda x, key : trace_loss_and_grad(x, na_weight, key), lens_desc=lens_desc, niters=niters, lr=lr)

    outputs = [run_opt_with_na_weight(lens_init, nw) for nw in na_weights_normalized]
    other_info_hist = [list_of_dict_transpose(output[2]) for output in outputs]

    # run optimization with bias gradient
    loss_params_bias = loss_params_uniform.copy()
    loss_params_bias['loss_type'] = 'spot_error'
    bias_loss = create_tracing_loss(lens_init, l2_loss_fn, **loss_params_uniform)
    bias_loss_and_grad = jax.jit(jax.value_and_grad(bias_loss, has_aux=True))
    output_bias = run_opt(lens_init, lambda x, key : bias_loss_and_grad(x, 0.0, key), lens_desc=lens_desc, niters=niters, lr=lr)
    bias_best_idx = np.argmin(output_bias[0])
    bias_best_state = output_bias[1][bias_best_idx]


    state_name_pairs = []
    for output, na in zip(outputs, na_weights):
        best_idx = np.argmin(output[0])
        best_state = output[1][best_idx]
        state_name_pairs.append((best_state, f"spot_{na}"))

    state_name_pairs.append((bias_best_state, f"spot_bias"))
    state_name_pairs.append((lens_init, f"spot_init"))

    lens_list, names = zip(*state_name_pairs)
    plot_spot_diagrams(lens_list, names, 20, 10, max_aperture)