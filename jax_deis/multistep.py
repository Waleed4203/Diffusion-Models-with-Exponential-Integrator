"""
jax_deis/multistep.py  —  AO-DEIS multi-order coefficient computation

Changes vs. original:
  • get_ab_eps_coef_all_orders()  — new function that precomputes coefficient
    tables for ALL polynomial orders r = 0..max_order at once.
  • ab_step_with_error()          — new function that returns BOTH the
    high-order and low-order estimates (for embedded error calculation).
  • ab_step() and get_ab_eps_coef() are fully backward-compatible.
"""

import jax
import jax.numpy as jnp

from .sde import MultiStepSDE

# ---------------------------------------------------------------------------
# Low-level quadrature helpers (unchanged)
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
    prod_{k != j}  (tau - t_{i+k}) / (t_{i+j} - t_{i+k})
    """
    num   = t_val - ts_poly
    denum = ts_poly[coef_idx] - ts_poly
    num   = num.at[coef_idx].set(1.0)
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
            num_item,
        )
        rtn = rtn.at[:order + 1].set(coef)
        return rtn
    return _worker


# ---------------------------------------------------------------------------
# Original single-order coefficient builders (unchanged / backward-compat)
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

    prev_coef = get_ab_eps_coef(
        sde, highest_order, timesteps[:order + 1], order=order - 1
    )
    cur_coef_worker = get_coef_per_step_fn(sde, highest_order, order)

    col_idx = jnp.arange(len(timesteps) - order - 1)[:, None]
    idx = col_idx + jnp.arange(order + 1)[None, :]
    vec_ts_poly = timesteps[idx]

    cur_coef = jax.vmap(cur_coef_worker, (0, 0, 0), 0)(
        timesteps[order:-1], timesteps[order + 1:], vec_ts_poly
    )
    return jnp.concatenate([prev_coef, cur_coef], axis=0)


# ---------------------------------------------------------------------------
# AO-DEIS: coefficient table for ALL orders at once
# ---------------------------------------------------------------------------

def get_ab_eps_coef_all_orders(sde, max_order, timesteps):
    """
    Precompute coefficient tables for ALL polynomial orders r = 0..max_order.

    Returns:
        coef_table: list of (max_order+1) JAX arrays, each shape
                    (num_step, max_order+2), where the first column is the
                    x-coefficient (ψ ratio) and the remaining columns are
                    the ε-polynomial coefficients for that order.
    """
    assert isinstance(sde, MultiStepSDE)
    x_coef = sde.psi(timesteps[:-1], timesteps[1:])   # (num_step,)

    coef_table = []
    for r in range(max_order + 1):
        eps_coef_r = get_ab_eps_coef(sde, max_order, timesteps, order=r)
        ab_coef_r  = jnp.concatenate([x_coef[:, None], eps_coef_r], axis=1)
        coef_table.append(ab_coef_r)

    return coef_table   # list[max_order+1] of jnp arrays (num_step, max_order+2)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def ab_step(x, ei_coef, new_eps, eps_pred):
    """
    Original single-order step (JAX version — eps_pred is a JAX array).
    Returns: (new_x, updated_eps_buffer)
    """
    x_coef, eps_coef = ei_coef[0], ei_coef[1:]
    full_eps = jnp.concatenate([new_eps[None], eps_pred])
    eps_term = jnp.einsum("i,i...->...", eps_coef, full_eps)
    return x_coef * x + eps_term, full_eps[:-1]


def ab_step_with_error(x, ei_coef_high, ei_coef_low, new_eps, eps_pred):
    """
    AO-DEIS embedded-error step (JAX version).

    Computes BOTH the high-order (r) and low-order (r-1) estimates and
    returns the embedded error scalar E_i = ||x_high - x_low||₂.

    Args:
        x:             Current state JAX array
        ei_coef_high:  Coefficient row for order r   (max_order+2,)
        ei_coef_low:   Coefficient row for order r-1 (max_order+2,)
        new_eps:       ε-prediction at the current timestep
        eps_pred:      JAX array of previous ε-predictions (max_order, ...)

    Returns:
        new_x       – next state using the high-order update
        new_eps_pred – updated ε history buffer
        error       – scalar L2 error norm (Python float)
    """
    # High-order update
    x_high, new_eps_pred = ab_step(x, ei_coef_high, new_eps, eps_pred)

    # Low-order update (same eps_pred, different coefficients)
    x_low, _ = ab_step(x, ei_coef_low, new_eps, eps_pred)

    # Embedded error scalar
    error = float(jnp.linalg.norm((x_high - x_low).reshape(-1)))

    return x_high, new_eps_pred, error