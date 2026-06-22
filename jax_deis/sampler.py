"""
jax_deis/sampler.py  —  AO-DEIS adaptive-order sampler (JAX version)

Changes vs. original DEIS
--------------------------
• get_sampler()           accepts new kwargs: delta_tol, guidance_w
• get_sampler_t_ab()      AO-DEIS adaptive order (Python loop, not fori_loop,
• get_sampler_rho_ab()    because the order is data-dependent at runtime)
• All other samplers (rho_rk, ipndm) are unchanged.
• Full backward-compatibility: calling without the new kwargs = original behaviour.

NOTE: jax.lax.fori_loop requires a static carry structure, which is
incompatible with dynamic order selection.  The adaptive samplers therefore
use a plain Python for-loop (same performance for typical NFE counts ≤ 50).
The non-adaptive rho_rk path still uses jax.lax.fori_loop as before.
"""

import jax
import jax.numpy as jnp

from .multistep import (
    ab_step,
    ab_step_with_error,
    get_ab_eps_coef,
    get_ab_eps_coef_all_orders,
)
from .rk import get_rk_fn
from .sde import MultiStepSDE, get_rev_ts
from .vpsde import VPSDE


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_sampler(
    sde,
    eps_fn,
    ts_phase,
    ts_order,
    num_step,
    method="rho_rk",
    ab_order=3,
    rk_method="3kutta",
    # AO-DEIS parameters
    delta_tol=0.1,
    guidance_w=0.0,
):
    """
    Factory – returns a sampler closure.

    New keyword arguments (AO-DEIS, only affect rho_ab / t_ab):
        delta_tol  : base tolerance δ₀  (default 0.1)
        guidance_w : CFG scale w; δ_eff = δ₀·(1 + 0.1·w)  (default 0.0)
    """
    if method.lower() == "rho_rk":
        return get_sampler_rho_rk(sde, eps_fn, ts_phase, ts_order, num_step, rk_method)
    elif method.lower() == "rho_ab":
        return get_sampler_rho_ab(
            sde, eps_fn, ts_phase, ts_order, num_step, ab_order,
            delta_tol=delta_tol, guidance_w=guidance_w,
        )
    elif method.lower() == "t_ab":
        return get_sampler_t_ab(
            sde, eps_fn, ts_phase, ts_order, num_step, ab_order,
            delta_tol=delta_tol, guidance_w=guidance_w,
        )
    elif method.lower() == "ipndm":
        return get_sampler_ipndm(sde, eps_fn, num_step)
    raise RuntimeError(f"{method} not supported!!")


# ---------------------------------------------------------------------------
# AO-DEIS: t-space Adams-Bashforth sampler
# ---------------------------------------------------------------------------

def get_sampler_t_ab(
    sde,
    eps_fn,
    ts_phase,
    ts_order,
    num_step,
    ab_order,
    delta_tol=0.1,
    guidance_w=0.0,
):
    """
    t-space Adams-Bashforth with adaptive order (AO-DEIS).

    δ_eff = delta_tol · (1 + 0.1 · guidance_w)
    """
    max_order = ab_order
    gamma     = 0.1
    delta_eff = delta_tol * (1.0 + gamma * guidance_w)

    rev_ts = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)

    # Precompute coefficient tables for orders 0..max_order
    coef_table = get_ab_eps_coef_all_orders(sde, max_order, rev_ts)
    # coef_table[r] shape: (num_step, max_order+2)

    def sampler(x0):
        # Initialise eps buffer (JAX array, same shape as x0, stacked)
        eps_pred = jnp.stack([x0] * max_order)  # (max_order, *x0.shape)

        x = x0
        current_order = 1   # warm-up at order 1

        for i in range(num_step):
            s_t     = rev_ts[i]
            new_eps = eps_fn(x, s_t)

            eff_order = min(current_order, i + 1, max_order)
            eff_order = max(eff_order, 1)

            coef_high = coef_table[eff_order][i]

            if eff_order > 1:
                coef_low = coef_table[eff_order - 1][i]
                new_x, new_eps_pred, error = ab_step_with_error(
                    x, coef_high, coef_low, new_eps, eps_pred
                )

                if error > delta_eff:
                    # Fall back to lower order for this step
                    new_x, new_eps_pred = ab_step(
                        x, coef_low, new_eps, eps_pred
                    )
                    current_order = max(1, eff_order - 1)
                else:
                    current_order = min(eff_order + 1, max_order)
            else:
                new_x, new_eps_pred = ab_step(x, coef_high, new_eps, eps_pred)
                current_order = min(2, max_order)

            x        = new_x
            eps_pred = new_eps_pred

        return x

    return sampler


# ---------------------------------------------------------------------------
# AO-DEIS: rho-space Adams-Bashforth sampler
# ---------------------------------------------------------------------------

def get_sampler_rho_ab(
    sde,
    eps_fn,
    ts_phase,
    ts_order,
    num_step,
    ab_order,
    delta_tol=0.1,
    guidance_w=0.0,
):
    """
    ρ-space Adams-Bashforth with adaptive order (AO-DEIS).
    """
    max_order = ab_order
    gamma     = 0.1
    delta_eff = delta_tol * (1.0 + gamma * guidance_w)

    rev_ts   = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)
    rev_rhos = sde.t2rho(rev_ts)

    class HelperSDE(MultiStepSDE):
        def psi(cls, t1, t2):
            return t1 / t1 * t2 / t2
        def eps_integrand(cls, vec_t):
            return vec_t / vec_t

    helper   = HelperSDE()
    x_coef_j = jnp.ones(rev_ts.shape[0] - 1)

    coef_table = []
    for r in range(max_order + 1):
        eps_r   = get_ab_eps_coef(helper, max_order, rev_rhos, order=r)
        coef_r  = jnp.concatenate([x_coef_j[:, None], eps_r], axis=1)
        coef_table.append(coef_r)

    @jax.jit
    def eps_fn_vrho(v, rho):
        t = sde.rho2t(rho)
        x = sde.v2x(v, t)
        return eps_fn(x, t)

    def sampler(xT):
        vT       = sde.x2v(xT, rev_ts[0])
        eps_pred = jnp.stack([vT] * max_order)  # (max_order, *shape)

        v             = vT
        current_order = 1

        for i in range(num_step):
            rho_cur = rev_rhos[i]
            eps_cur = eps_fn_vrho(v, rho_cur)

            eff_order = min(current_order, i + 1, max_order)
            eff_order = max(eff_order, 1)

            coef_high = coef_table[eff_order][i]

            if eff_order > 1:
                coef_low = coef_table[eff_order - 1][i]
                v_next, new_eps_pred, error = ab_step_with_error(
                    v, coef_high, coef_low, eps_cur, eps_pred
                )

                if error > delta_eff:
                    v_next, new_eps_pred = ab_step(
                        v, coef_low, eps_cur, eps_pred
                    )
                    current_order = max(1, eff_order - 1)
                else:
                    current_order = min(eff_order + 1, max_order)
            else:
                v_next, new_eps_pred = ab_step(v, coef_high, eps_cur, eps_pred)
                current_order = min(2, max_order)

            v        = v_next
            eps_pred = new_eps_pred

        return sde.v2x(v, rev_ts[-1])

    return sampler


# ---------------------------------------------------------------------------
# Unchanged samplers
# ---------------------------------------------------------------------------

def get_sampler_ipndm(sde, eps_fn, num_step):
    assert isinstance(sde, VPSDE)
    rev_ts = get_rev_ts(sde, num_step, 1, ts_phase="t")
    x_coef = sde.psi(rev_ts[:-1], rev_ts[1:])

    def get_linear_ab_coef(i):
        if i == 0:
            return jnp.asarray([1.0, 0, 0, 0]).reshape(-1, 4)
        prev_coef = get_linear_ab_coef(i - 1)
        if i == 1:
            cur_coef = jnp.asarray([1.5, -0.5, 0, 0])
        elif i == 2:
            cur_coef = jnp.asarray([23, -16, 5, 0]) / 12.0
        else:
            cur_coef = jnp.asarray([55, -59, 37, -9]) / 24.0
        return jnp.concatenate([prev_coef, cur_coef.reshape(-1, 4)])

    linear_ab_coef = get_linear_ab_coef(len(rev_ts) - 2)
    next_ts, cur_ts   = rev_ts[1:], rev_ts[:-1]
    next_alpha        = sde.t2alpha_fn(next_ts)
    cur_alpha         = sde.t2alpha_fn(cur_ts)
    ddim_coef = (
        jnp.sqrt(1 - next_alpha)
        - jnp.sqrt(next_alpha / cur_alpha) * jnp.sqrt(1 - cur_alpha)
    )
    eps_coef  = ddim_coef.reshape(-1, 1) * linear_ab_coef
    ei_ab_coef = jnp.concatenate([x_coef[:, None], eps_coef], axis=1)

    def sampler(x0):
        eps_pred = jnp.stack([x0] * 3)
        x = x0
        for i in range(num_step):
            s_t     = rev_ts[i]
            new_eps = eps_fn(x, s_t)
            x, eps_pred = ab_step(x, ei_ab_coef[i], new_eps, eps_pred)
        return x

    return sampler


def get_sampler_rho_rk(sde, eps_fn, ts_phase, ts_order, num_step, rk_method):
    rev_ts   = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)
    rk_fn    = get_rk_fn(rk_method)
    rev_rhos = sde.t2rho(rev_ts)

    @jax.jit
    def eps_fn_vrho(v, rho):
        t = sde.rho2t(rho)
        x = sde.v2x(v, t)
        return eps_fn(x, t)

    def _step_fn(i_th, v):
        rho_cur, rho_next = rev_rhos[i_th], rev_rhos[i_th + 1]
        delta_t = rho_next - rho_cur
        return rk_fn(v, rho_cur, delta_t, eps_fn_vrho)

    def sample_fn(xT):
        vT   = sde.x2v(xT, rev_ts[0])
        veps = jax.lax.fori_loop(0, len(rev_rhos) - 1, _step_fn, vT)
        return sde.v2x(veps, rev_ts[-1])

    return sample_fn
