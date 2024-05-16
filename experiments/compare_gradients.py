import jax
import jax.numpy as jnp
import tqdm
import argparse
import matplotlib.pyplot as plt


from dlt import render
from dlt import zemax_loader

SPP=10
RESOLUTION=50
NPASSES = 100


def test_images(init_rng_seed, render_curvature):
    npasses = 100
    spp = 10
    sens_height = 1.0
    focal_length = 50.0
    imheight = 1.2 * sens_height / jnp.sqrt(sens_height**2 + focal_length**2)
    imresh = 100
    imresw = 100

    im_outname = f'image_{init_rng_seed}'
    imeps_outname = f'imageeps_{init_rng_seed}'
    grad_nowarp_outname = f'grad_nowarp_{init_rng_seed}'
    grad_outname = f'ad_grad_{init_rng_seed}'
    fd_outname = f'fd_grad_{init_rng_seed}'

    usaf_sampler = render.image_sampler('data/siggraphlogo_bw.jpg', imheight, imheight)
    sens = render.QuantizedRect(sens_height, imresh, sens_height, imresw, 10.0, jnp.pi/8)

    lens = zemax_loader.load_zemax_file('data/zemax_files/double-gauss-50mm.zmx')
    lens_init = jnp.array(lens)

    init_rng_key = jax.random.PRNGKey(init_rng_seed)

    if render_curvature:
        k_input = lens_init[1, 0]
    else:
        # start the aperture at smaller value so that there aren't any complex effects
        # finite difference has a lot of artifacting and aliasing at maximum aperture
        k_input = lens_init[6, 3] * 0.75 

    def get_lens(k0):
        if render_curvature:
            lens_run = lens_init.at[1, 0].set(k0)
        else:
            lens_run = lens_init.at[6, 3].set(k0) 
        return lens_run

    def render_fun_warp(k0, rngkey):
        lens_run = get_lens(k0)
        im, valid_ratio = render.render_zemax_image_farfield(lens_run, sens, usaf_sampler, spp=spp, rng_key=rngkey, nbatches=1, verbose=False, nowarp=False)
        return im, (im, valid_ratio)

    def render_fun_nowarp(k0, rngkey):
        lens_run = get_lens(k0)
        im, valid_ratio = render.render_zemax_image_farfield(lens_run, sens, usaf_sampler, spp=spp, rng_key=rngkey, nbatches=1, verbose=False, nowarp=True)
        return im, (im, valid_ratio)

    render_fun_jit = jax.jit(render_fun_warp)
    render_fun_nowarp_jit = jax.jit(render_fun_nowarp)
    grad_fun = jax.jit(jax.jacfwd(render_fun_jit, has_aux=True))
    grad_nowarp_fun = jax.jit(jax.jacfwd(render_fun_nowarp_jit, has_aux=True))

    k0 = k_input

    print('rendering aperture aware gradient')
    rng_key = init_rng_key
    grad_im = 0
    for i in tqdm.tqdm(range(npasses)):
        rng_key, subkey = jax.random.split(rng_key)
        grad_im_cur, (im, valid_ratio) = grad_fun(k0, subkey)
        grad_im = grad_im + grad_im_cur / npasses
    jnp.save(grad_outname, grad_im)

    print('rendering aperture unaware gradient')
    rng_key = init_rng_key
    grad_nowarp_im = 0
    for i in tqdm.tqdm(range(npasses)):
        rng_key, subkey = jax.random.split(rng_key)
        grad_im_cur, (im, valid_ratio) = grad_nowarp_fun(k0, subkey)
        grad_nowarp_im = grad_nowarp_im + grad_im_cur / npasses
    jnp.save(grad_nowarp_outname, grad_nowarp_im)

    print('rendering image and image epsilon')
    rng_key = init_rng_key
    full_im = 0
    full_im_eps = 0
    eps = 1e-6 if render_curvature else 1e-2
    for i in tqdm.tqdm(range(npasses)):
        rng_key, subkey = jax.random.split(rng_key)

        im, aux = render_fun_nowarp_jit(k0, subkey)
        full_im = full_im + im / npasses

        im_eps, aux = render_fun_nowarp_jit(k0 + eps, subkey)
        full_im_eps = full_im_eps + im_eps / npasses

    fd_grad = (full_im_eps - full_im) / eps
    jnp.save(im_outname, full_im)
    jnp.save(imeps_outname, full_im_eps)
    jnp.save(fd_outname, fd_grad)

    outnames = [im_outname, imeps_outname, grad_nowarp_outname, grad_outname, fd_outname]
    images = [full_im, full_im_eps, grad_nowarp_im, grad_im, fd_grad]

    for outname, image in zip(outnames, images):
        plt.figure()
        plt.imshow(image)
        plt.title(outname)
        plt.savefig(f'{outname}.png')
        plt.close()


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_disable_jit", True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--rng_seed', type=int, default=0)
    parser.add_argument('--curvature', action='store_true')
    args = parser.parse_args()
    print(args)

    test_images(args.rng_seed, args.curvature)
