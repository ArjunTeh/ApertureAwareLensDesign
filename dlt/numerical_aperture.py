import jax
import jax.numpy as jnp

from . import aspheric
from . import curvature_sphere
from . import tracing
from . import zemax


def bisection_search(fn, low, hi, max_iter=30, tol=1e-8):
    '''Find a root of the function fn using bisection search.'''
    lo_pass, lo_aux = fn(low) 
    hi_pass, hi_aux = fn(hi)

    # if not lo_pass:
    #     raise ValueError("low val does not pass")

    # if hi_pass:
    #     raise ValueError("hi val passes")

    def cond_fun(carry):
        iter_num, lo_arg, hi_arg, aux = carry
        close = jnp.abs(hi_arg - lo_arg) < tol
        maxed_out = iter_num >= max_iter
        return jnp.logical_not(jnp.logical_or(close, maxed_out))

    def body_fun(carry):
        iter_num, lo_arg, hi_arg, aux = carry
        mid_arg = (lo_arg + hi_arg) / 2
        mid_pass, mid_aux = fn(mid_arg)
        lo_arg = jnp.where(mid_pass, mid_arg, lo_arg)
        hi_arg = jnp.where(mid_pass, hi_arg, mid_arg)
        aux = jnp.where(mid_pass, mid_aux, aux)
        return iter_num + 1, lo_arg, hi_arg, aux

    init = (0, low, hi, lo_aux) 
    iter_num, lo_arg, hi_arg, aux = jax.lax.while_loop(cond_fun, body_fun, init)
    return lo_arg, (iter_num, aux)


def calc_na(surfs, fn_suite):
    '''Calculate the numerical aperture of a stack of surfaces'''
    max_ap = jnp.array(surfs[0][1]) * 2 # for optical axis, first surface is a limiter
    # max_ap = jnp.array([s[1] for s in surfs[:-1]]).max() * 2

    def check_ray(val):
        x = jnp.array([-1.0, val, 0.0])
        v = jnp.array([1.0, 0., 0.])
        L = jnp.array(1.0)
        # valid, xp, vp, Lp = tracing.trace(implicits, constraints, surfs, x, v, L)
        (valid, xp, vp, Lp), warpfield = tracing.trace_scan_no_stop(fn_suite, surfs, x, v, L)
        return valid, xp

    boundary, aux = bisection_search(check_ray, 0.0, max_ap)
    return boundary, aux


def calc_zemax_na_angle(state, angle, stop_id):
    surf_funs = curvature_sphere.functionSuite() if state.shape[1] == 4 else aspheric.functionSuite()
    surfs = zemax.zemax_state_to_jax_surfaces(state)

    def check_ray(val):
        x = jnp.array([-1.0, val, 0.0])
        v = jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
        v = v / jnp.linalg.norm(v)
        L = jnp.array(1.0)

        (valid, xp, vp, Lp), (Tx, wargmax) = tracing.trace_scan_with_stop_check(surf_funs, surfs, x, v, L, stop_id)
        return valid, wargmax

    max_val = state[:, 3].max() * 2
    hi_bound, aux = bisection_search(check_ray, 0.0001, max_val)
    lo_bound, aux = bisection_search(check_ray, -0.0001, -max_val)
    return (lo_bound, hi_bound), aux


def calc_zemax_na(state, asphere=False):
    if asphere:
        surf_funs = aspheric.functionSuite()
    else:
        surf_funs = curvature_sphere.functionSuite()

    surfs = zemax.zemax_state_to_jax_surfaces(state)

    return calc_na(surfs, surf_funs)


def retract_to_pupil_size(throughput_fn, state, direction):
    '''Retract the lens state to a given pupil size'''

    def body_fun(carry):
        iter_num, val, cur_step  = carry
        new_state = state + cur_step * direction
        val, state_grad = throughput_fn(new_state)
        val_grad = jnp.sum(state_grad * direction) 
        next_step = cur_step - val / val_grad
        return iter_num + 1, val, next_step

    def cond_fun(carry):
        iter_num, val, cur_step = carry
        return jnp.logical_and((iter_num < 100), (jnp.abs(val) > 1e-4))

    iters, val, step = jax.lax.while_loop(cond_fun, body_fun, (0, 1.0, 0.0))

    return state + step * direction


def na_gradient_2d(state, angle, stop_id):
    '''Calculate the gradient of the numerical aperture with respect to the lens state'''
    surf_funs = curvature_sphere.functionSuite() if state.shape[1] == 4 else aspheric.functionSuite()
    imfun, confun, (warpfun, warpgrad) = surf_funs

    boundary, aux = calc_zemax_na_angle(state, angle, stop_id)
    fail_surf_id = aux[1]
    x = jnp.array([-1.0, boundary[0], 0.0])
    v = jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
    L = jnp.array(1.0)

    def min_constraint(state, fail_id):
        surfs = zemax.zemax_state_to_jax_surfaces(state)
        (valid, xp, vp, Lp), (Tx, wargmin) = tracing.trace_scan_with_stop_check(surf_funs, surfs, x, v, L, fail_surf_id)
        xf = xp[fail_id]
        vf = vp[fail_id-1]
        Vx, (posg, velg) = warpfun(surfs[fail_id], xf, vf)
        return -2 * jnp.pi * jnp.maximum(posg, velg)

    grad_fun = jax.grad(min_constraint)
    return grad_fun(state, fail_surf_id)
