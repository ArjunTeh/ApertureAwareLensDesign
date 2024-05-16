import jax
import jax.numpy as jnp

from functools import partial


@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def _rad_index(x0, height, width, hres, wres, bottom_left):
    # assume that the point is already on the sensor plane
    x_sens = x0
    xn1 = (x_sens[1] - bottom_left[1]) * hres / height
    xn2 = (x_sens[2] - bottom_left[2]) * wres / width
    xn = jnp.array([xn1, xn2])

    offset = jnp.mgrid[:2, :2].reshape((2, -1))
    # idx00 = jnp.floor(jnp.array([xn1, xn2])).astype(int)
    idx00 = jnp.ceil(jnp.array([xn1, xn2]) - 1).astype(int)

    idx = idx00[:, None] + offset
    return xn, idx


def sensor_splat(image, imsize, bottom_left, pos, L):
    """Splat points into an optical axis aligned image using a kernel.

    Args:
        image (np.ndarray): The image to splat points into.
        points (np.ndarray): The point to splat into the image.

    Returns:
        jnp.ndarray: The image with points splatted onto it.
    """
    # Get the shape of the image
    height, width = imsize
    hres, wres = image.shape
    x0 = pos

    # idx = N x 2 x 4
    # xn = N x 2
    xn, idx = _rad_index(x0, height, width, hres, wres, bottom_left)

    # tent filter
    # w = N x 4
    # L1 tent
    # w = jnp.clip(1 - (jnp.abs(xn[:, :, None] - idx - 0.5)).prod(axis=1), a_min=0, a_max=1)

    # L2 tent
    xn_l2dist = jnp.sqrt(((xn[:, :, None] - idx - 0.5)**2).sum(axis=1))
    w = jnp.clip(1 - xn_l2dist, a_min=0, a_max=1)

    wsum = w.sum(axis=-1, keepdims=True)
    wsum = jnp.where(jnp.isclose(wsum, 0), 1, wsum)

    val = w * L[:, None] / wsum

    image_splatted = image.at[idx[:, 0, :], idx[:, 1, :]].add(val, mode="drop")
    return image_splatted
