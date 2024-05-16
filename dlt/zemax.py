import jax
import jax.numpy as jnp

from . import curvature_sphere
from . import aspheric
from . import constants
from . import sources


def zemax_state_to_jax_surfaces(state, scene_scale=None, reparameterize=True, focal_length=None, object_distance=jnp.inf):
    if scene_scale is None:
        scene_scale = jnp.max(jax.lax.stop_gradient(state[:, 3]))
        # scene_scale = 1.0
    jax_surfs = []
    # ior_prev = constants.DEFAULT_EXT_IOR
    ior_prev = state[-1, 2] # the medium between the last surface and the sensor is air
    cur_pos = 0
    for s in state:
        if s.shape[0] == 4:
            surf = curvature_sphere.curvature_sphere(
                curvature=s[0],
                origin=cur_pos,
                aperture=s[3],
                ior_in=ior_prev,
                ior_out=s[2]
            )
        elif s.shape[0] == 13:
            if reparameterize:
                surf = aspheric.asphere(
                    curvature=s[0],
                    origin=cur_pos,
                    aperture=s[3],
                    ior_in=ior_prev,
                    ior_out=s[2],
                    higher_order_terms=s[4:],
                    scene_scale=scene_scale
                )
            else:
                surf = aspheric.asphere_normalized(
                    curvature=s[0],
                    origin=cur_pos,
                    aperture=s[3],
                    ior_in=ior_prev,
                    ior_out=s[2],
                    higher_order_terms=s[4:],
                    scene_scale=scene_scale
                )
        else:
            raise ValueError("Invalid surface shape")
        ior_prev = s[2]
        cur_pos = cur_pos + s[1]
        jax_surfs.append(surf)


    if focal_length is None:
        sensor_pos = cur_pos
    else:
        elp = cur_pos - focal_length
        u = jnp.abs(object_distance + elp)
        v = 1 / ((1 / focal_length) - (1 / u))
        sensor_pos = elp + v

    # add one last surface that represents the sensor plane
    # TODO(ateh): this should probably be a different kind of object
    if state.shape[-1] == 4:
        sensor = curvature_sphere.curvature_sphere(
            curvature=0.,
            origin=sensor_pos,
            aperture=constants.FLOAT_BIG_NUMBER,
            ior_in=constants.DEFAULT_EXT_IOR,
            ior_out=constants.DEFAULT_EXT_IOR
        )
    else:
        sensor = aspheric.asphere(
            curvature=0.0,
            origin=sensor_pos,
            aperture=constants.FLOAT_BIG_NUMBER,
            ior_in=constants.DEFAULT_EXT_IOR,
            ior_out=constants.DEFAULT_EXT_IOR,
            higher_order_terms=jnp.zeros_like(state[0, 4:]),
            scene_scale=scene_scale
        )
    jax_surfs.append(sensor)

    return jax_surfs


def normalize_zemax_to_asphere(asphere_state, scene_scale):
    '''Normalize the asphere coefficients to the same scale as the specified aperture'''
    data = asphere_state.copy()
    ho_scale = (scene_scale**2) ** jnp.arange(1, 9)
    scene_scale2 = scene_scale**2

    data[:, 0] = asphere_state[:, 0] * scene_scale2 # curvature
    data[:, 4] = (1 + asphere_state[:, 4]) / scene_scale2 # conic section
    data[:, 5:13] = asphere_state[:, 5:13] * ho_scale[None, :] # higher order terms

    return data


def normalize_asphere_to_zemax(asphere_state, scene_scale):
    '''Convert the asphere coefficients to its original scale, specified by scene_scale'''
    data = asphere_state.copy()
    ho_scale = (scene_scale**2) ** jnp.arange(1, 9)
    scene_scale2 = scene_scale**2

    data[:, 0] = asphere_state[:, 0] / scene_scale2 # curvature
    data[:, 4] = (asphere_state[:, 4] * scene_scale2) - 1 # conic section
    data[:, 5:13] = asphere_state[:, 5:13] / ho_scale[None, :] # higher order terms

    return data


def reparameterize_zemax_to_asphere(asphere_state, scene_scale):
    '''Reparameterize the higher order coefficients to the same scale as the specified aperture'''
    data = jnp.asarray(asphere_state)
    powers = jnp.arange(1, 9)
    higher_order_terms = asphere_state[:, 5:13]
    higher_order_sign = jnp.sign(higher_order_terms)

    higher_order_terms = higher_order_sign * jnp.abs(higher_order_terms) ** (1/powers[None, :])

    data = data.at[:, 5:13].set(higher_order_terms)

    return data


def reparameterize_asphere_to_zemax(asphere_state, scene_scale):
    '''Reparameterize the higher order coefficients to the same scale as the specified aperture'''
    data = jnp.asarray(asphere_state)
    powers = jnp.arange(1, 9)
    higher_order_terms = asphere_state[:, 5:13]
    ho_sign = jnp.sign(higher_order_terms)

    higher_order_terms = ho_sign * jnp.abs(higher_order_terms) ** (powers[None, :])
    data = data.at[:, 5:13].set(higher_order_terms)

    return data
