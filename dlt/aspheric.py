import jax
import jax.numpy as jnp

from functools import partial

from . import constants
from . import surface_base


def asphere(curvature, origin, higher_order_terms, aperture, ior_in, ior_out, scene_scale=1.0):
    '''Create an aspherical surface with the given parameters
    The higher order terms are in the order of the Zemax asphere coefficients

    Data goes in the order:
        origin, aperture, ior_in, ior_out, curvature, scene_scale, higher_order_terms
    '''
    # ho_scale = (scene_scale**2) ** jnp.arange(0, 9)
    data = jnp.zeros(1+1+2+1+1+9) # number of parameters
    data = data.at[0].set(origin)
    data = data.at[1].set(aperture)
    data = data.at[2].set(ior_in)
    data = data.at[3].set(ior_out)
    data = data.at[4].set(scene_scale)
    data = data.at[5].set(curvature)
    data = data.at[6:15].set(higher_order_terms)

    return reparameterize_asphere(data, scene_scale)


def normalize_asphere(asphere_surf, scene_scale):
    '''Normalize the asphere coefficients to the same scale as the specified aperture'''
    ho_scale = (scene_scale**2) ** jnp.arange(1, 9)
    curvature = asphere_surf[5]
    conic = asphere_surf[6]
    higher_order_terms = asphere_surf[7:16]

    data = jnp.zeros(1+1+2+1+1+9) # number of parameters
    data = data.at[0].set(asphere_surf[0]) # origin
    data = data.at[1].set(asphere_surf[1]) # aperture
    data = data.at[2].set(asphere_surf[2]) # ior_in
    data = data.at[3].set(asphere_surf[3]) # ior_out
    data = data.at[4].set(scene_scale) # scene_scale
    data = data.at[5].set(curvature * scene_scale * scene_scale) # curvature
    data = data.at[6].set((conic + 1) / scene_scale / scene_scale) # conic
    data = data.at[7:15].set(higher_order_terms * ho_scale) # higher_order_terms
    return data


def reparameterize_asphere(asphere_surf, scene_scale):
    '''Reparameterize the higher order coefficients to the same scale as the specified aperture'''
    powers = jnp.arange(1, 9)
    bb_data = asphere_surf[0:4]
    old_scene_scale = asphere_surf[4, None]
    curvature = asphere_surf[5, None]
    conic = asphere_surf[6, None]
    higher_order_terms = asphere_surf[7:15]
    higher_order_sign = jnp.sign(higher_order_terms)
    higher_order_terms = higher_order_sign * jnp.abs(higher_order_terms) ** (1/powers)

    data = jnp.concatenate([bb_data, jnp.array([scene_scale]), curvature, conic, higher_order_terms])
    return data


def asphere_normalized(curvature, origin, higher_order_terms, aperture, ior_in, ior_out, scene_scale=1.0):
    '''Create an aspherical surface with the given parameters
    The higher order terms are in the order of the Zemax asphere coefficients

    Data goes in the order:
        origin, aperture, ior_in, ior_out, curvature, scene_scale, higher_order_terms
    '''
    ho_scale = (scene_scale**2) ** jnp.arange(1, 9)
    data = jnp.zeros(1+1+2+1+1+9) # number of parameters
    data = data.at[0].set(origin)
    data = data.at[1].set(aperture)
    data = data.at[2].set(ior_in)
    data = data.at[3].set(ior_out)
    data = data.at[4].set(scene_scale)
    data = data.at[5].set(curvature)
    data = data.at[6].set(higher_order_terms[0])
    data = data.at[7:15].set(higher_order_terms[1:] * ho_scale)
    return data


def _asphere_value(surface, r2):
    k = surface[5]
    C = surface[6]
    scene_scale2 = surface[4]**2

    r2clamp = jnp.minimum(r2, surface[1]**2)
    recip_sqrt_term = 1 - (1 + C) * k**2 * r2clamp
    bad_mask = recip_sqrt_term < 0
    recip_sqrt_term = jnp.where(bad_mask, 1.0, recip_sqrt_term)

    init_val = k * r2 / (1 + jnp.sqrt(recip_sqrt_term))

    val = init_val
    for i in range(8):
        exponent = i + 1
        val = val + surface[i + 7] * ((r2/scene_scale2)**exponent)

    return val


def implicitAsphericalReparameterized_valonly(surface, x):
    '''First component is the OA'''
    origin = surface[0]
    semidiam = surface[1]
    scene_scale = surface[4]
    scene_scale2 = scene_scale**2
    xn = (x - jnp.array([origin, 0., 0.]))
    r2 = jnp.dot(xn[1:], xn[1:]) 
    r2clamp = jnp.minimum(r2, semidiam**2)

    k = surface[5]
    C = surface[6]
    sqrt_term = 1 - (1 + C) * k**2 * r2clamp
    bad_mask = sqrt_term < 0
    sqrt_term = jnp.where(bad_mask, 1.0, sqrt_term)
    init_val = k * r2 / (1 + jnp.sqrt(sqrt_term))

    val = init_val
    for i in range(8):
        exponent = i + 1
        coeff = surface[i + 7]
        val = val + jnp.sign(coeff) * ((jnp.abs(coeff) * r2clamp / scene_scale2)**exponent)

    val = jnp.where(bad_mask, 0, val)
    val = val - xn[0]
    return val


def implicitAspherical_valonly(surface, x):
    '''First component is the OA'''
    origin = surface[0]
    semidiam = surface[1]
    xn = (x - jnp.array([origin, 0., 0.]))
    r2 = jnp.dot(xn[1:], xn[1:])
    r2 = jnp.minimum(r2, semidiam**2)

    point = _asphere_value(surface, r2)
    val = point - xn[0]
    return val


def implicitAspherical(surface, x):
    g, gx = jax.value_and_grad(implicitAspherical_valonly, 1)(surface, x)
    # g, gx = jax.value_and_grad(implicitAsphericalNormalized_valonly, 1)(surface, x)
    return g, gx


def implicitAsphericalFactory(asphere_type):
    # asphere_type = 'standard'
    if asphere_type == 'reparameterized':
        fun = implicitAsphericalReparameterized_valonly
    elif asphere_type == 'standard':
        fun = implicitAspherical_valonly
    else:
        raise ValueError(f'Unknown asphere type {asphere_type}')

    return jax.value_and_grad(fun, 1)

def functionSuite(asphere_type='standard'):
    imp_asphere = implicitAsphericalFactory(asphere_type)
    constraintFullAspherical = surface_base.constraint_aperture
    warpfieldAspherical = partial(surface_base.warpfield, imp_asphere)
    warp_grad = surface_base.warpfield_grad_mat_generator(imp_asphere)
    return imp_asphere, constraintFullAspherical, (warpfieldAspherical, warp_grad)