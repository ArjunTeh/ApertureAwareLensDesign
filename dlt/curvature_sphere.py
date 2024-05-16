import jax
import jax.numpy as jnp

from functools import partial

from . import surface_base


def curvature_sphere(curvature=0., origin=0., aperture=10000., ior_in=1.0, ior_out=1.5):
    return jnp.array([origin, aperture, ior_in, ior_out, curvature])


def implicitCurvatureSphere(surface, x):
    '''First component is the OA'''
    k = surface[-1]
    origin = surface[0]
    xn = x - jnp.array([origin, 0., 0.])
    g =  k * xn[0]**2 + k * xn[1]**2 + k * xn[2]**2 - 2 * xn[0]
    gx = 2 * k * xn - jnp.array([2, 0, 0])
    return g, gx


def functionSuite():
    constraintCurvatureSphere = surface_base.constraint_aperture
    warpfieldCurvatureSphere = partial(surface_base.warpfield, implicitCurvatureSphere)
    warp_grad = surface_base.warpfield_grad_mat_generator(implicitCurvatureSphere)
    return implicitCurvatureSphere, constraintCurvatureSphere, (warpfieldCurvatureSphere, warp_grad)
