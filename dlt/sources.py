import jax
import jax.numpy as jnp

import numpy as np


@jax.jit
def total_focal_distance(effective_focal_length, dist_to_object):
    '''1 / f = 1 / u + 1 / v'''
    sensor_diop = 1 / effective_focal_length - 1 / dist_to_object
    return (1 / sensor_diop) + dist_to_object


@jax.jit
def vec_from_angle_2d(angle):
    return jnp.stack([jnp.cos(angle), jnp.sin(angle)])


@jax.jit
def angle_from_vec_2d(vec):
    return jnp.arctan(vec[1] / vec[0])


def hat_box_sample(N, angle=jnp.pi, key=jax.random.PRNGKey(0)):
    '''Returns rays sampled such that the first coordinate is the major direction
    Can also specify an angle
    '''
    key, subkey1, subkey2 = jax.random.split(key, 3)

    minx = jnp.cos(angle)
    x = jax.random.uniform(subkey1, shape=(N,), minval=minx, maxval=1.0)

    scale = jnp.sqrt(1 - x**2)
    theta = jax.random.uniform(subkey2, shape=(N,), minval=0.0, maxval=2*jnp.pi)
    y = scale*jnp.sin(theta)
    z = scale*jnp.cos(theta)

    return jnp.stack([x, y, z], axis=-1)



def point_source_quadrature(half_length, angle, distance):
    height = jnp.tan(angle) * distance
    pos = jnp.array([-distance, -height, 0.0])
    top_ray = jnp.array([0.0, half_length, 0.0]) - pos
    bot_ray = jnp.array([0.0, -half_length, 0.0]) - pos
    top_ang = angle_from_vec_2d(top_ray)
    bot_ang = angle_from_vec_2d(bot_ray)
    
    def generate_rays(nrays, key=None):
        angs = jnp.linspace(bot_ang, top_ang, nrays)

        x = jnp.tile(pos[None, :], (nrays, 1))
        v = jnp.stack([jnp.cos(angs), jnp.sin(angs), jnp.zeros(nrays)], axis=1)
        rad = jnp.ones((x.shape[0],))

        t_to_src_pln = (distance - 5.0) / v[:, 0]
        x = x + t_to_src_pln[:, None] * v
        weights = np.ones_like(rad)
        return x, v, rad, weights

    return generate_rays


def line_source_quadrature(half_length, angle):
    def generate_rays(nrays, key=None):
        x = jnp.linspace(-half_length, half_length, nrays)
        y = jnp.zeros_like(x)
        z = -5 * jnp.ones_like(x)
        center = jnp.sin(angle) * -5

        pos = jnp.stack([z, x + center, y], axis=1)
        vel = jnp.tile(np.stack([np.cos(angle), np.sin(angle), 0.0]), (pos.shape[0], 1))
        rad = jnp.ones((pos.shape[0],))
        weights = jnp.ones_like(rad)
        return pos, vel, rad, weights, x

    return generate_rays


def line_source_stratified_random(half_length, angle):
    def generate_rays(nrays, key):
        dx = (jax.random.uniform(key, (nrays,)) - 0.5) * half_length * 2 / nrays
        x = jnp.linspace(-half_length, half_length, nrays) + dx
        y = jnp.zeros_like(x)
        z = -5 * jnp.ones_like(x)
        center = jnp.sin(angle) * -5
        x_rad = jnp.cos(angle) * x

        pos = jnp.stack([z, x + center, y], axis=1)
        vel = jnp.tile(jnp.stack([jnp.cos(angle), jnp.sin(angle), 0.0]), (pos.shape[0], 1))
        rad = jnp.ones((pos.shape[0],))
        weights = jnp.ones_like(rad)
        return pos, vel, rad, weights, x_rad

    return generate_rays


def line_source_stratified_random_angle(half_length):
    def generate_rays(nrays, angle, key):
        dx = (jax.random.uniform(key, (nrays,)) - 0.5) * half_length * 2 / nrays
        x = jnp.linspace(-half_length, half_length, nrays) + dx
        y = jnp.zeros_like(x)
        z = -5 * jnp.ones_like(x)
        center = jnp.sin(angle) * -5
        x_rad = jnp.cos(angle) * x

        pos = jnp.stack([z, x + center, y], axis=1)
        vel = jnp.tile(jnp.stack([jnp.cos(angle), jnp.sin(angle), 0.0]), (pos.shape[0], 1))
        rad = jnp.ones((pos.shape[0],))
        weights = jnp.ones_like(rad)
        return pos, vel, rad, weights, x_rad

    return generate_rays


def line_source_stratified_sampling():
    def generate_rays(nrays, half_length, angle, key):
        dx = (jax.random.uniform(key, (nrays,)) - 0.5) * half_length * 2 / nrays
        x = jnp.linspace(-half_length, half_length, nrays) + dx
        y = jnp.zeros_like(x)
        z = -5 * jnp.ones_like(x)
        center = jnp.sin(angle) * -5
        x_rad = jnp.cos(angle) * x

        pos = jnp.stack([z, x + center, y], axis=1)
        vel = jnp.tile(jnp.stack([jnp.cos(angle), jnp.sin(angle), 0.0]), (pos.shape[0], 1))
        rad = jnp.ones((pos.shape[0],))
        weights = jnp.ones_like(rad)
        return pos, vel, rad, weights, jnp.abs(x_rad)
    return generate_rays


def square_source_quadrature(width, height=None, angle=None):
    if height is None:
        height = width

    def generate_rays(nrays, key):
        Xm, Ym = jnp.meshgrid(jnp.linspace(-width/2, width/2, nrays), 
                             jnp.linspace(-height/2, height/2, nrays))
        x, y = Xm.flatten(), Ym.flatten()
        z = -jnp.zeros_like(x)

        pos = jnp.stack([z, x, y], axis=1)
        vel = jnp.tile(jnp.stack([np.cos(angle), -jnp.sin(angle), 0.0]), (pos.shape[0], 1))
        rad = jnp.ones((pos.shape[0],))
        pos = pos - vel * 1.0 / vel[:, 0:1]
        weights = jnp.ones_like(rad)
        xradius = jnp.ones_like(pos[:, 0])

        return pos, vel, rad, weights, xradius

    return generate_rays


def square_source_random_stratified(width, height=None, angle=None):
    if height is None:
        height = width

    def generate_rays(nrays, key):
        dx = (jax.random.uniform(key, (2, nrays*nrays)) - 0.5) * width / nrays
        Xm, Ym = jnp.meshgrid(jnp.linspace(-width/2, width/2, nrays), 
                              jnp.linspace(-height/2, height/2, nrays))
        x, y = Xm.flatten(), Ym.flatten()
        z = -np.ones_like(x)

        pos = np.stack([z, x, y], axis=1)
        vel = np.tile(np.stack([1.0, 0.0, 0.0]), (pos.shape[0], 1))
        rad = np.ones((pos.shape[0],))
        return pos, vel, rad

    return generate_rays


def square_source_random(width, height=None, angle=None):
    if height is None:
        height = width

    def generate_rays(nrays):
        x = np.random.uniform(-width/2, width/2, nrays)
        y = np.random.uniform(-height/2, height/2, nrays)
        z = -np.ones_like(x)

        pos = np.stack([z, x, y], axis=1)
        vel = np.tile(np.stack([1.0, 0.0, 0.0]), (pos.shape[0], 1))
        rad = np.ones((pos.shape[0],))
        weights = np.ones_like(rad)
        return pos, vel, rad

    return generate_rays


def square_point_source(scale, position, direction, up):

    t1 = np.cross(direction, up)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(direction, t1)
    t2 = t2 / np.linalg.norm(t2)

    def generate_rays(nrays):
        Xm, Ym = np.meshgrid(np.linspace(-scale/2, scale/2, nrays), 
                             np.linspace(-scale/2, scale/2, nrays))
        offset_pos = (
            Xm.flatten()[:, None] * t1[None, :] + 
            Ym.flatten()[:, None] * t2[None, :] +
            direction[None, :]
        )

        vel = offset_pos / np.linalg.norm(offset_pos, axis=-1, keepdims=True)
        pos = np.tile(position[None, :], (vel.shape[0], 1))
        rad = np.ones((vel.shape[0],))
        return pos, vel, rad

    return generate_rays


def square_point_random(nrays, position, direction, up, cone_angle=jnp.pi/4, rng_key=jax.random.PRNGKey(0)):
    t1 = jnp.cross(direction, up)
    t1 = t1 / jnp.linalg.norm(t1)
    t2 = jnp.cross(direction, t1)
    t2 = t2 / jnp.linalg.norm(t2)

    v0 = hat_box_sample(nrays, cone_angle, key=rng_key)
    vel = (
        v0[:, 2:3] * t1[None, :] + 
        v0[:, 1:2] * t2[None, :] +
        v0[:, 0:1] * direction[None, :]
    )

    vel = vel / jnp.linalg.norm(vel, axis=-1, keepdims=True)
    pos = jnp.tile(position[None, :], (vel.shape[0], 1))
    rad = jnp.ones((vel.shape[0],))
    return pos, vel, rad


def leggaus_circular_quadrature(radius, rings, angles, src_angle=0.0, object_distance=np.inf):
    '''infinite conjugate only? '''
    x, w = np.polynomial.legendre.leggauss(rings)
    
    rho = np.sqrt((1 + x) / 2) * radius
    weights = w / 2

    pos_list = []
    ang_list = np.linspace(0, 2*np.pi, angles + 1)[:-1]
    for a in ang_list:
        pos_list.append(np.stack([np.zeros_like(rho), np.cos(a) * rho, np.sin(a) * rho], axis=1))

    lens_pos = np.concatenate(pos_list, axis=0)
    if object_distance == np.inf:
        vel = np.tile(np.array([np.cos(src_angle), np.sin(src_angle), 0.0]), (lens_pos.shape[0], 1))
    else:
        object_pos = np.array([-object_distance, -object_distance * np.tan(src_angle), 0.0])
        vel_un = lens_pos - object_pos
        vel = vel_un / np.linalg.norm(vel_un, axis=-1, keepdims=True)
    
    pos = lens_pos - vel * 5.0 / vel[:, 0:1]
    rad = np.ones((pos.shape[0],))
    x_rad = np.tile(rho, (angles,))

    weights = np.tile(weights, (angles,))

    def generate_rays(nrays, key):
        return pos, vel, rad, weights, x_rad
    return generate_rays


def target_positions(nangles, target_focal_length):
    target_heights = [target_focal_length * jnp.tan(ang) for ang in nangles]
    return target_heights


def create_multipoint_source(line_range, focal_length, object_distance=jnp.inf, angle_list=[0.0], source_type='line', stochastic=False, key=jax.random.PRNGKey(0)):
    angles = angle_list
    if stochastic:
        line_generators = [line_source_stratified_random(line_range, a) for a in angles]
    else:
        if source_type == 'line':
            if object_distance == jnp.inf:
                line_generators = [line_source_quadrature(line_range, a) for a in angles]
            else:
                line_generators = [point_source_quadrature(line_range, a, object_distance) for a in angles]
        elif source_type == 'leggauss':
            line_generators = [leggaus_circular_quadrature(line_range, 6, 4, a) for a in angles]
        elif source_type == 'rectangle':
            line_generators = [square_source_quadrature(line_range, line_range, a) for a in angles]
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

        return x, v, L, jnp.asarray(targets), w, jnp.abs(rad), line_range

    return generate_rays