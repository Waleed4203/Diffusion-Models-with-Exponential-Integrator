import jax
import jax.numpy as jnp

from .sde import MultiStepSDE

# ---------------------------------------------------------------------------
# Low-level quadrature helpers (unchanged from original DEIS)
# ---------------------------------------------------------------------------

def get_integrator_basis_fn(sde):
    def _worker(t_start, t_end, num_item):
        dt = (t_end - t_start) / num_item
        t_inter = jnp.linspace(t_start, t_end, num_item, endpoint=False)
        psi_coef = sde.psi(t_inter, t_end)
        integrand = sde.eps_integrand(t_inter)
        return psi_coef * integrand, t_inter, dt
    return _worker


def single_poly_coef(t_val, ts_poly, coef_idx=0):
    """
    prod_{k != j} (tau - t_{i+k}) / (t_{i+j} - t_{i+k})
    t_val: tau
    ts_poly: t_{i+k}
    j: coef_idx
    """
    num = t_val - ts_poly
    denum = ts_poly[coef_idx] - ts_poly
    num = num.at[coef_idx].set(1.0)
    denum = denum.at[coef_idx].set(1.0)
    return jnp.prod(num) / jnp.prod(denum)

vec_poly_coef = jax.vmap(single_poly_coef, (0, None, None), 0)


def get_one_coef_per_step_fn(sde):
    _eps_coef_worker_fn = get_integrator_basis_fn(sde)
    def _worker(t_start, t_end, ts_poly, coef_idx=0, num_item=10000):
        """C_{ij}  (j = coef_idx)"""
        integrand, t_inter, dt = _eps_coef_worker_fn(t_start, t_end, num_item)
        poly_coef = vec_poly_coef(t_inter, ts_poly, coef_idx)
        return jnp.sum(integrand * poly_coef) * dt
    return _worker


def get_coef_per_step_fn(sde, highest_order, order):
    eps_coef_fn = get_one_coef_per_step_fn(sde)
    def _worker(t_start, t_end, ts_poly, num_item=10000):
        """C_i  (flipped j ordering)"""
        rtn = jnp.zeros((highest_order + 1,), dtype=float)
        ts_poly = ts_poly[:order + 1]
        coef = jax.vmap(eps_coef_fn, (None, None, None, 0, None))(
            t_start, t_end, ts_poly,
            jnp.flip(jnp.arange(order + 1)),
            num_item
        )
        rtn = rtn.at[:order + 1].set(coef)
        return rtn
    return _worker


# ---------------------------------------------------------------------------
# Original single-order coefficient builders (unchanged)
# ---------------------------------------------------------------------------

def get_ab_eps_coef_order0(sde, highest_order, timesteps):
    _worker = get_coef_per_step_fn(sde, highest_order, 0)
    col_idx = jnp.arange(len(timesteps) - 1)[:, None]
    idx = col_idx + jnp.arange(1)[None, :]
    vec_ts_poly = timesteps[idx]
    return jax.vmap(_worker, (0, 0, 0), 0)(
        timesteps[:-1], timesteps[1:], vec_ts_poly
    )


def get_ab_eps_coef(sde, highest_order, timesteps, order):
    """Return coefficient table for a single fixed order (original API)."""
    assert isinstance(sde, MultiStepSDE)
    if order == 0:
        return get_ab_eps_coef_order0(sde, highest_order, timesteps)

    prev_coef = get_ab_eps_coef(sde, highest_order, timesteps[:order + 1], order=order - 1)
    cur_coef_worker = get_coef_per_step_fn(sde, highest_order, order)

    col_idx = jnp.arange(len(timesteps) - order - 1)[:, None]
    idx = col_idx + jnp.arange(order + 1)[None, :]
    vec_ts_poly = timesteps[idx]

    cur_coef = jax.vmap(cur_coef_worker, (0, 0, 0), 0)(
        timesteps[order:-1], timesteps[order + 1:], vec_ts_poly
    )
    return jnp.concatenate([prev_coef, cur_coef], axis=0)


# ---------------------------------------------------------------------------
# AO-DEIS: multi-order coefficient table
# ---------------------------------------------------------------------------

def get_ab_eps_coef_all_orders(sde, max_order, timesteps):
    """
    Precompute coefficient tables for ALL polynomial orders r = 0, 1, …, max_order.

    Returns:
        coef_table: list of length (max_order + 1), where
                    coef_table[r] has shape (num_step, max_order + 2)
                    = [x_coef | eps_coef_r0, ..., eps_coef_r_maxorder]
                      (same layout as the original ab_coef but for order r)

    At runtime the sampler indexes into coef_table[current_order][step_i].
    """
    assert isinstance(sde, MultiStepSDE)
    n_steps = len(timesteps) - 1

    x_coef = sde.psi(timesteps[:-1], timesteps[1:])   # (n_steps,)

    coef_table = []
    for r in range(max_order + 1):
        eps_coef_r = get_ab_eps_coef(sde, max_order, timesteps, order=r)
        # Shape: (n_steps, max_order+1)
        ab_coef_r = jnp.concatenate([x_coef[:, None], eps_coef_r], axis=1)
        # Shape: (n_steps, max_order+2)
        coef_table.append(ab_coef_r)

    return coef_table  # list[max_order+1] of (n_steps, max_order+2)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def ab_step(x, ei_coef, new_eps, eps_pred):
    """
    Original single-order step (th_deis uses Python lists for eps_pred).
    Returns: (new_x, updated_eps_buffer)
    """
    x_coef, eps_coef = ei_coef[0], ei_coef[1:]
    full_eps_pred = [new_eps, *eps_pred]
    rtn = x_coef * x
    for cur_coef, cur_eps in zip(eps_coef, full_eps_pred):
        rtn += cur_coef * cur_eps
    return rtn, full_eps_pred[:-1]


def ab_step_with_error(x, ei_coef_high, ei_coef_low, new_eps, eps_pred):
    """
    AO-DEIS adaptive step.

    Computes BOTH the high-order (r) and low-order (r-1) estimates and
    returns the embedded error norm E_i = ||x_high - x_low||_2.

    Args:
        x:            Current state  (torch Tensor)
        ei_coef_high: Coefficient row for order r   shape (max_order+2,)
        ei_coef_low:  Coefficient row for order r-1 shape (max_order+2,)
        new_eps:      eps prediction at current step
        eps_pred:     list of previous eps predictions (length max_order)

    Returns:
        new_x:        x_{i+1} using the high-order update
        new_eps_pred: updated eps buffer
        error:        scalar L2 norm of (x_high - x_low)
    """
    # High-order update
    x_high, new_eps_pred = ab_step(x, ei_coef_high, new_eps, eps_pred)

    # Low-order update (uses the same eps_pred buffer)
    x_low, _ = ab_step(x, ei_coef_low, new_eps, eps_pred)

    # Embedded error: L2 norm of the difference (flattened)
    diff = x_high - x_low
    # Flatten everything after the batch dim for norm calculation
    import torch
    error = diff.norm().item() if hasattr(diff, 'norm') else float(
        jnp.linalg.norm(diff.reshape(-1))
    )

    return x_high, new_eps_pred, error
