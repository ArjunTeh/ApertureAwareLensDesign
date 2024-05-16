from functools import partial
import jax
import jax.numpy as jnp

from . import constants
from . import surface_base
from . import optical_properties


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def intersectNewton(implicit_fun, state, x, v, t0):
    t0 = t0.copy()

    def body_fun(carry):
        td_old, td_new, iter_num = carry
        t = t0 + td_new
        xn = x + t*v
        num, imgrad = implicit_fun(state, xn)
        den = jnp.dot(imgrad, v)
        tupdate = num / den
        return td_new, td_new - tupdate, iter_num+1

    def cond_fun(carry):
        td_old, td_new, iter_num = carry

        not_close_enough = jnp.abs(td_old - td_new) > constants.NEWTON_EPSILON
        return jnp.logical_and(not_close_enough, iter_num < constants.NEWTON_MAX_ITER)

    g, gx = implicit_fun(state, x + t0*v)
    tprev, tstar, iter_num = jax.lax.while_loop(cond_fun, body_fun, (0., -g / jnp.dot(gx, v), 0))
    tnew = t0 + tstar
    failed = jnp.logical_or(tnew < 0, iter_num >= constants.NEWTON_MAX_ITER)
    return jnp.logical_not(failed), x + tnew * v, tnew

@intersectNewton.defjvp
def intersectNewton_jvp(implicit_fun, primals, tangents):
    state, x, v, t0 = primals
    dstate, dx, dv, dt0 = tangents

    success, xstar, tstar = intersectNewton(implicit_fun, state, x, v, t0)

    (g, gx), (jvp_g, hvp_g) = jax.jvp(lambda s : implicit_fun(s, xstar), (state,), (dstate,))

    dtdx = -jnp.dot(gx, dx) / jnp.dot(v, gx)
    dtdv = -tstar * jnp.dot(gx, dv) / jnp.dot(v, gx)
    dtdp = - jvp_g / jnp.dot(gx, v)
    t_tangent = dtdx + dtdv + dtdp

    dxdx = dx + v * dtdx
    dxdv = tstar * dv + v * dtdv
    dxdp = v * dtdp
    x_tangent = dxdx + dxdv + dxdp

    return (success, xstar, tstar), (success, x_tangent, t_tangent)


def refract(implicit_fun, state, x, v):
    g, gx = implicit_fun(state, x)
    normal = gx / jnp.linalg.norm(gx)

    # rays canonically go from left to right
    # n_left should be the medium that the ray is coming from
    # n_right is the medium that the ray will be exiting into
    n_left = state[2]
    n_right = state[3]

    cos_ang = jnp.dot(-v, normal)
    eta = jnp.where(cos_ang < 0, n_right / n_left, n_left / n_right)

    # check for TIR
    cos_crit = jnp.sqrt(jnp.maximum(1 - (1 / eta)**2, constants.FLOAT_EPSILON))
    failed = jnp.abs(cos_ang) < cos_crit

    k = jnp.maximum(1 - (eta**2 * (1 - cos_ang**2)), constants.FLOAT_EPSILON)
    T = eta * v + ((eta * jnp.abs(cos_ang) - jnp.sqrt(k)) * jnp.sign(cos_ang) * normal)
    return jnp.logical_not(failed), T / jnp.linalg.norm(T)


def radiance(implicit_fun, state, x, v0, v1, L):
    g, gx = implicit_fun(state, x)
    normal = gx / jnp.linalg.norm(gx)

    n_i = state[2]
    n_o = state[3]

    cos_i = jnp.abs(jnp.dot(normal, v0))
    cos_o = jnp.abs(jnp.dot(normal, v1))

    spolar_num = n_o*cos_i - n_i*cos_o
    spolar_den = n_o*cos_i + n_i*cos_o
    ppolar_num = n_o*cos_o - n_i*cos_i
    ppolar_den = n_o*cos_o + n_i*cos_i

    fs = spolar_num / spolar_den
    fp = ppolar_num / ppolar_den

    fresnel_term = 0.5 * (fs**2 + fp**2)

    fresnel_thru = (1 - fresnel_term) * (n_o**2 / n_i**2)

    return fresnel_thru * L


def trace_by_scan(surf_funcs, surfs, x, v, L):
    # the rays will traces through everything with no regard for constraints
    # a ray is invalid if it is unable to intersect an object or it traces the objects
    # in not the right order.
    valid = jnp.array(True)
    imfun, confun, (warpfun, warpgrad) = surf_funcs

    def loop_fun(carry, s):
        valid, x0, v0, L0 = carry

        t0 = (s[0] - x0[0]) / v0[0]
        intersect_success, x1, t1 = intersectNewton(imfun, s, x0, v0, t0)

        # check for success
        valid = jnp.logical_and(valid, intersect_success)

        refract_success, v1 = refract(imfun, s, x1, v0)
        valid = jnp.logical_and(valid, refract_success)

        L1 = radiance(imfun, s, x0, v0, v1, L0)

        return (valid, x1, v1, L1), (x1, v1, L1)

    (valid, xt, vt, Lt), (xp, vp, Lp) = jax.lax.scan(loop_fun, (valid, x, v, L), jnp.stack(surfs))

    return valid, jnp.concatenate([x[None, :], xp], axis=0), jnp.concatenate([v[None, :], vp], axis=0), jnp.concatenate([L[None], Lp], axis=0)


def calculate_warpfields(surf_funcs, surfs, xp, vp, Lp):
    imfun, confun, (warpfun, warpgrad) = surf_funcs

    surf_stack = jnp.stack(surfs)
    x1 = xp[1:]
    v0 = vp[:-1]

    # check for surface intersections 
    # and replace the diameter with the intersection height
    surf_semidiam = surf_stack[:, 1]
    thickness = surf_stack[1:, 0] - surf_stack[:-1, 0]
    
    if surf_stack.shape[1] < 6: # working with spherical surfaces
        # curvature surfaces
        max_heights = optical_properties.effective_semidiameter(surf_stack[:-1, 4], surf_stack[1:, 4], thickness)
        semidiams = jnp.minimum(surf_semidiam[:-1], max_heights)
        surf_stack = surf_stack.at[:-1, 1].set(semidiams)
        semidiams = jnp.minimum(surf_semidiam[1:], max_heights)
        surf_stack = surf_stack.at[1:, 1].set(semidiams)

    warpgrad_v = jax.vmap(warpgrad, (0, 0, 0))
    posg, velg, warp_ap, warp_tir = warpgrad_v(surf_stack, x1, v0) 
    return posg, warp_ap, velg, warp_tir


def trace_scan_with_warpfield(surf_funcs, surfs, x, v, L):
    valid, xp, vp, Lp = trace_by_scan(surf_funcs, surfs, x, v, L)
    posg, warp_ap, velg, warp_tir = calculate_warpfields(surf_funcs, surfs, xp, vp, Lp)
    
    # tracer doesn't account for constraints
    valid = jnp.logical_and(valid, jnp.all(posg <= 0.0))
    valid = jnp.logical_and(valid, jnp.all(velg <= 0.0))

    Tx, wargmin = mix_warpfields(posg[:-1], warp_ap[:-1], velg[:-1], warp_tir[:-1])

    return (valid, xp[-1], vp[-1], Lp[-1]), (Tx, wargmin)


def trace_scan_no_stop(surf_funcs, surfs, x, v, L, stop_id=0):
    imfun, confun, (warpfun, warpgrad) = surf_funcs
    valid, xp, vp, Lp = trace_by_scan(surf_funcs, surfs, x, v, L)

    posg, warp_ap, velg, warp_tir = calculate_warpfields(surf_funcs, surfs, xp, vp, Lp)

    # check the stop surface after the fact
    valid = jnp.logical_and(valid, jnp.all(posg <= 0.0))
    valid = jnp.logical_and(valid, jnp.all(velg <= 0.0))

    Tx, wargmin = mix_warpfields(posg[:-1], warp_ap[:-1], velg[:-1], warp_tir[:-1])
    return (valid, xp, vp, Lp), (Tx, wargmin)

def trace_scan_no_stop_data(surf_funcs, surfs, x, v, L, stop_id=0):
    imfun, confun, (warpfun, warpgrad) = surf_funcs
    valid, xp, vp, Lp = trace_by_scan(surf_funcs, surfs, x, v, L)

    posg, warp_ap, velg, warp_tir = calculate_warpfields(surf_funcs, surfs, xp, vp, Lp)

    # check the stop surface after the fact
    valid = jnp.logical_and(valid, jnp.all(posg <= 0.0))
    valid = jnp.logical_and(valid, jnp.all(velg <= 0.0))

    return (valid, xp, vp, Lp), (posg, velg)


def trace_scan_with_stop_check(surf_funcs, surfs, x, v, L, stop_idx):
    '''Here we assume that there is only one stop in the system
        The aperture stop can be something that can be behind other elements in the system.
        So we check the aperture after the fact rather than during tracing in order to not break
        the ordering of the elements.
    '''
    imfun, confun, (warpfun, warpgrad) = surf_funcs
    surfs_no_stop = surfs[:stop_idx] + surfs[stop_idx+1:]
    valid, xp, vp, Lp = trace_by_scan(surf_funcs, surfs_no_stop, x, v, L)

    posg, warp_ap, velg, warp_tir = calculate_warpfields(surf_funcs, surfs_no_stop, xp, vp, Lp)

    # check the stop surface after the fact
    stop_surf = surfs[stop_idx]
    xstop = xp[stop_idx] - vp[stop_idx] * (xp[stop_idx][0] - stop_surf[0]) / vp[stop_idx][0]
    gxstop, gvstop, warp_stopap, warp_stoptir = warpgrad(stop_surf, xstop, vp[stop_idx])

    valid = jnp.logical_and(valid, jnp.all(posg <= 0.0))
    valid = jnp.logical_and(valid, jnp.all(velg <= 0.0))
    valid = jnp.logical_and(valid, jnp.all(gxstop <= 0.0))
    valid = jnp.logical_and(valid, jnp.all(gvstop <= 0.0))

    Tx, wargmin = mix_warpfields(jnp.concatenate([posg[:stop_idx], gxstop[None], posg[stop_idx:-1]]), 
                                 jnp.concatenate([warp_ap[:stop_idx], warp_stopap[None], warp_ap[stop_idx:-1]]), 
                                 jnp.concatenate([velg[:stop_idx], gvstop[None], velg[stop_idx:-1]]), 
                                 jnp.concatenate([warp_tir[:stop_idx], warp_stoptir[None], warp_tir[stop_idx:-1]]))
    return (valid, xp, vp, Lp), (Tx, wargmin)


def max_warpfield(constraints_ap, warpfields_ap, constraints_tir, warpfields_tir):
    warg_ap = jnp.argmax(constraints_ap)
    warg_tir = jnp.argmax(constraints_tir)

    Vx_ap = warpfields_ap[warg_ap]
    Vx_tir = warpfields_tir[warg_tir]

    aperture_active = constraints_ap[warg_ap] >= constraints_tir[warg_tir]
    warg_min = jnp.where(aperture_active, warg_ap, -warg_tir)
    Vx = jnp.where(aperture_active, Vx_ap, Vx_tir)

    Tx = jnp.eye(6) + Vx - jax.lax.stop_gradient(Vx)
    return Tx, warg_min


def mix_warpfields(constraints_ap, warpfields_ap, constraints_tir, warpfields_tir):

    # calculate total warpfield / don't count the last surface
    gamma = 6
    def weighting_fn(g):
        w_attenuate = jnp.tanh(-jnp.abs(g) * 15) + 1
        return w_attenuate / jnp.maximum(jnp.abs(g)**gamma, constants.FLOAT_EPSILON)

    weights_ap = [weighting_fn(g) for g in constraints_ap]
    weights_tir = [weighting_fn(g) for g in constraints_tir]

    w_total = sum(weights_ap) + sum(weights_tir)
    w_total = jnp.maximum(w_total, constants.FLOAT_EPSILON)

    weights_norm_ap = jnp.array([w / w_total for w in weights_ap])
    weights_norm_tir = jnp.array([w / w_total for w in weights_tir])

    # total mixture
    Tx_total = 0
    for w, Vx in zip(weights_ap, warpfields_ap):
        Tx = (jnp.eye(6) + Vx - jax.lax.stop_gradient(Vx))
        Tx_total += Tx * (w / w_total)

    for w, Vx in zip(weights_tir, warpfields_tir):
        Tx = jnp.eye(6) + Vx - jax.lax.stop_gradient(Vx)
        Tx_total += Tx * (w / w_total)

    warg_ap = jnp.argmax(jnp.abs(weights_norm_ap))
    warg_tir = jnp.argmax(jnp.abs(weights_norm_tir))
    warg_max = jnp.where(weights_norm_ap[warg_ap] >= weights_norm_tir[warg_tir], warg_ap, -warg_tir)
    return Tx_total, warg_max
