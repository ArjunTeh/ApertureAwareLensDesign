import jax.numpy as jnp
import jax.scipy.optimize as jop
import jax
from tqdm import tqdm

from . import constants


def optimize_zemax_lens(loss_func, state):
    output = jop.minimize(loss_func, state, method='BFGS')
    return output
    

def finite_difference(loss_fn, state, eps=1e-6):
    orig_shape = state.shape
    stateravel = jnp.asarray(state.ravel())
    cur_loss = loss_fn(state)

    grad = jnp.zeros_like(stateravel)
    for i in range(stateravel.shape[0]):
        stateraveldx = stateravel.at[i].add(eps)
        dx_loss = loss_fn(stateraveldx.reshape(orig_shape))
        grad = grad.at[i].set((dx_loss - cur_loss) / eps)

    return grad.reshape(orig_shape)


def run_adam_descent(loss_fn, state, niters, termination_eps=None, constraint=None, projector=None, eps=1e-3, rng_key=None, beta1=0.9, beta2=0.999, mo1=None, mo2=None, show_progress=False, save_cadence=1, logger=None):
    '''returns loss trajectory, state trajectory, other info from the loss function, *adam info'''
    radii = state.copy()

    loss_hist = []
    grad_hist = []
    state_hist = []
    other_info_hist = []
    mo1_hist = []
    mo2_hist = []
    update_direction = []
    rng_key_hist = []

    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key

    mo1 = jnp.zeros_like(state) if mo1 is None else mo1
    mo2 = jnp.zeros_like(state) if mo2 is None else mo2
    for i in tqdm(range(niters), disable=(not show_progress)):
        if not jnp.all(jnp.isfinite(radii)):
            print("failed at iter", i)
            break

        rng_key, subkey = jax.random.split(rng_key)
        (loss_val, other_info), loss_grad = loss_fn(radii, subkey)

        if i == 0:
            prev_loss_val = loss_val

        if not jnp.all(jnp.isfinite(loss_grad)):
            print("calculated nan grad at iter", i)
            break
    
        if termination_eps is not None and (i > 0): 
            # Add epsilon termination condition
            # rel_diff = jnp.abs(loss_val - prev_loss_val) / jnp.abs(prev_loss_val)
            rel_diff = jnp.linalg.norm(loss_grad) / jnp.abs(prev_loss_val)
            if rel_diff < termination_eps:
                print("loss val", loss_val)
                print("grad norm", jnp.linalg.norm(loss_grad))
                print("prev loss val", prev_loss_val)
                print("relative difference", rel_diff)
                print("termination condition met at iter", i)
                break

        if i % save_cadence == 0:
            if logger is not None:
                logger(i, radii, loss_val, loss_grad, other_info)
            loss_hist.append(loss_val)
            grad_hist.append(loss_grad)
            state_hist.append(radii)
            other_info_hist.append(other_info)
            mo1_hist.append(mo1)
            mo2_hist.append(mo2)
            rng_key_hist.append(subkey)

        if projector is not None:
            loss_grad = projector(radii, loss_grad)

        # if jnp.any(loss_grad > 1.0):
        #     raise ValueError("loss grad is too large")

        mo1 = beta1*mo1 + loss_grad * (1 - beta1)
        mo2 = beta2*mo2 + (loss_grad**2) * (1 - beta2)

        mo1hat = mo1 / (1 - beta1**(i+1))
        mo2hat = mo2 / (1 - beta2**(i+1))

        update_dir = -eps * mo1hat / (jnp.sqrt(mo2hat) + constants.FLOAT_EPSILON)
        radii = radii + update_dir
        update_direction.append(update_dir)

        prev_loss_val = loss_val

        if constraint is not None:
            radii = constraint(radii)

    loss_hist.append(loss_val)
    grad_hist.append(loss_grad)
    state_hist.append(radii)
    other_info_hist.append(other_info)
    mo1_hist.append(mo1)
    mo2_hist.append(mo2)
    update_direction.append(update_dir)
    rng_key_hist.append(subkey)

    return loss_hist, state_hist, other_info_hist, grad_hist, mo1_hist, mo2_hist, update_direction, rng_key_hist


def gradient_descent(loss_fn, state, niters, constraint=None, manifold_fns=None, eps=1e-3, show_progress=False, save_cadence=1):
    '''returns loss trajectory, state trajectory, other info from the loss function'''
    radii = state.copy()

    if manifold_fns is not None:
        project_fn, retract_fn = manifold_fns
        radii = retract_fn(radii, radii)
    else:
        project_fn = lambda x, y : y
        retract_fn = lambda x, y : y

    loss_hist = []
    grad_hist = []
    state_hist = []
    orig_state_hist = []
    other_info_hist = []

    rng_key = jax.random.PRNGKey(0)

    for i in tqdm(range(niters), disable=(not show_progress)):
        assert jnp.all(jnp.isfinite(radii))

        rng_key, subkey = jax.random.split(rng_key)
        (loss_val, other_info), loss_grad = loss_fn(radii, subkey)

        assert jnp.all(jnp.isfinite(loss_grad))

        loss_grad = project_fn(radii, loss_grad)

        if i % save_cadence == 0:
            loss_hist.append(loss_val)
            grad_hist.append(loss_grad)
            state_hist.append(radii)
            other_info_hist.append(other_info)

        radii_step = radii - eps * loss_grad / jnp.sqrt(i+1)
        orig_state_hist.append(radii_step)
        radii = retract_fn(radii, radii_step)

        if constraint is not None:
            radii = constraint(radii)
    
    loss_hist.append(loss_val)
    grad_hist.append(loss_grad)
    state_hist.append(radii)
    other_info_hist.append(other_info)

    return loss_hist, state_hist, other_info_hist, grad_hist, orig_state_hist
