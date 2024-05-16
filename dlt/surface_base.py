import jax
import jax.numpy as jnp

from . import constants


def constraint_aperture(surface, x : jnp.ndarray):
    aperture = surface[1]

    xproj = x.at[0].set(0.0)

    x2 = jnp.dot(xproj, xproj)
    g = x2 / (aperture**2) - 1
    return g


# def constraint_aperture_and_grad2(implicit_fn, surface, x : jnp.ndarray):
#     aperture = surface[1]

#     # xproj = (x - jnp.array([surface[0], 0, 0])).at[0].set(0.)
#     xproj = x.at[0].set(0.0)

#     g = (jnp.linalg.norm(xproj) / aperture) - 1
#     gx = xproj / jnp.linalg.norm(xproj) / aperture
#     gv = jnp.zeros_like(x)

#     # g = jnp.dot(xproj, xproj) / aperture**2 - 1
#     # gx = 2*xproj / aperture**2
#     # gv = jnp.zeros_like(x)
#     return g, gx, gv

def constraint_aperture_and_grad(implicit_fn, surface, x : jnp.ndarray):
    g, gx = jax.value_and_grad(constraint_aperture, 1)(surface, x)
    gv = jnp.zeros_like(x)
    return g, gx, gv


def constraint_tir(implicit_fn, surface, x : jnp.ndarray, v : jnp.ndarray):
    '''Returns only the constraint'''
    g_grad = implicit_fn(surface, x)[1]
    # normal = g_grad / jnp.linalg.norm(g_grad)

    # rays canonically go from left to right
    # n_left should be the medium that the ray is coming from
    # n_right is the medium that the ray will be exiting into
    n_left = surface[2]
    n_right = surface[3]

    cos_ang_scaled = jnp.dot(-v, g_grad)
    eta = jnp.where(cos_ang_scaled < 0, n_right / n_left, n_left / n_right)

    # check for TIR
    cos_crit2 = 1 - (1 / eta)**2
    cos_ang2 = jnp.dot(v, g_grad)**2 / jnp.dot(g_grad, g_grad)

    # cos_crit2 = 1 - (surface[2] / surface[3])**2 # missing square root
    g = cos_crit2 - cos_ang2
    return g


def constraint_tir_and_grad(implicit_fn, surface, x : jnp.ndarray, v : jnp.ndarray):
    '''Returns the constraint and its gradient'''
    g, (gx, gv) = jax.value_and_grad(constraint_tir, (2, 3))(implicit_fn, surface, x, v)
    return g, gx, gv


def constraints(implicit_fn, surface, x : jnp.ndarray, v : jnp.ndarray):
    posg = constraint_aperture(surface, x)
    velg = constraint_tir(implicit_fn, surface, x, v)
    return posg, velg


def warpfield(implicit_fn, surface, x : jnp.ndarray, v : jnp.ndarray):
    # aperture warpfield
    posg = constraint_aperture(surface, x)
    posg_, posgx_, posgv_ = constraint_aperture_and_grad(implicit_fn, jax.lax.stop_gradient(surface), x)
    posg_norm2_ = jnp.dot(posgx_, posgx_)
    posg_norm2_ = jnp.maximum(posg_norm2_, constants.FLOAT_EPSILON)
    Vx_aperture = -posgx_ * posg  / posg_norm2_

    # TIR warpfield
    velg = constraint_tir(implicit_fn, surface, x, v)
    velg_, velgx_, velgv_ = constraint_tir_and_grad(implicit_fn, jax.lax.stop_gradient(surface), x, v)
    vel_norm2 = jnp.dot(velgx_, velgx_) + jnp.dot(velgv_, velgv_)
    vel_norm2 = jnp.maximum(vel_norm2, constants.FLOAT_EPSILON)

    Vx_tir = -velgx_ * velg  / vel_norm2
    Vv_tir = -velgv_ * velg / vel_norm2

    valid = posg_ < 0
    valid = jnp.logical_and(valid, velg_ < 0)

    Vx_aperture = jnp.where(valid, Vx_aperture, 0.0)
    Vx_tir = jnp.where(valid, Vx_tir, 0.0)
    Vv_tir = jnp.where(valid, Vv_tir, 0.0)

    return (Vx_aperture, Vx_tir, Vv_tir), (posg, velg)


def warpfield_from_constraint_fun(constraint_fun, surfaces, xp, vp, Lp):
    g, gx, gv = constraint_fun(surfaces, xp, vp, Lp)
    g_, gx_, gv_ = constraint_fun(jax.lax.stop_gradient(surfaces), xp, vp, Lp)

    g_norm2_ = jnp.dot(gx_, gx_) + jnp.dot(gv_, gv_)
    g_norm2_ = jnp.maximum(g_norm2_, constants.FLOAT_EPSILON)
    Vx = -gx_ * g  / g_norm2_
    Vv = -gv_ * g  / g_norm2_

    valid = g_ < 0
    Vx = jnp.where(valid, Vx, 0.0)
    Vv = jnp.where(valid, Vv, 0.0)

    return (Vx, Vv), g


def warpfield_grad(implicit_fn, surface, x : jnp.ndarray, v : jnp.ndarray):
    warpfield_fn = lambda s, x, v: warpfield(implicit_fn, s, x, v)
    warp_grad = jax.jacfwd(warpfield_fn, (1, 2))
    return warp_grad


def warpfield_grad_mat_generator(implicit_fn):
    warpfield_fn = lambda s, x, v: warpfield(implicit_fn, s, x, v)
    warp_grad = jax.jacfwd(warpfield_fn, (1, 2), has_aux=True)

    def warp_grad_mat(s, x, v):
        (Vx_aperture, Vx_tir, Vv_tir), (posg, velg) = warp_grad(s, x, v)
        warp_ap = jnp.zeros((6, 6))
        warp_ap = warp_ap.at[:3, :3].set(Vx_aperture[0])
        warp_ap = warp_ap.at[:3, 3:].set(Vx_aperture[1])

        warp_tir = jnp.zeros((6, 6))
        warp_tir = warp_tir.at[:3, :3].set(Vx_tir[0])
        warp_tir = warp_tir.at[:3, 3:].set(Vx_tir[1])
        warp_tir = warp_tir.at[3:, :3].set(Vv_tir[0])
        warp_tir = warp_tir.at[3:, 3:].set(Vv_tir[1])

        valid = jnp.logical_and(posg < 0, velg < 0)
        warp_ap = jnp.where(valid, warp_ap, 0.0)
        warp_tir = jnp.where(valid, warp_tir, 0.0)

        return posg, velg, warp_ap, warp_tir

    return warp_grad_mat
