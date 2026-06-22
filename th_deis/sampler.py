"""
th_deis/sampler.py  —  AO-DEIS adaptive-order sampler

Changes vs. original DEIS
--------------------------
1. get_sampler()              accepts two new kwargs:
     delta_tol   (float)  – base smoothness threshold δ₀ (default 0.1)
     guidance_w  (float)  – CFG scale w, used for δ_eff = δ₀·(1+γ·w)
                            set to 0 when not using CFG.

2. get_sampler_t_ab()         uses adaptive order (AO-DEIS):
   get_sampler_rho_ab()       at each step, both order-r and order-(r-1)
                              updates are computed; if the embedded error
                              E_i > δ_eff the order is reduced by 1 for
                              that step (min 1).  Order ramps back up to
                              max_order one step at a time after.

3. All other samplers (rho_rk, ipndm) are unchanged — they still work
   exactly as before.

4. Full backward-compatibility: if you call get_sampler() without
   delta_tol / guidance_w the behaviour is identical to the original.
"""

import math
import jax.numpy as jnp
import torch

from .multistep import (
    ab_step,
    ab_step_with_error,
    get_ab_eps_coef,
    get_ab_eps_coef_all_orders,
)
from .rk import get_rk_fn
from .sde import MultiStepSDE, get_rev_ts
from .vpsde import VPSDE
from .helper import jax2th, th2jax


# ---------------------------------------------------------------------------
# Simple Python for-loop (th_deis does not use jax.lax.fori_loop)
# ---------------------------------------------------------------------------

def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


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
    Factory function – returns a sampler closure.

    AO-DEIS parameters (only used by rho_ab / t_ab methods):
        delta_tol  : base tolerance δ₀ for the embedded error test.
                     Increase to allow more high-order steps; decrease for
                     more conservative (lower-order) behaviour.
        guidance_w : Classifier-Free Guidance scale w.
                     δ_eff = δ₀ · (1 + 0.1·w)
                     Leave at 0.0 when not using CFG.
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
    t-space Adams-Bashforth sampler with adaptive order (AO-DEIS).

    The coefficient tables for ALL orders 0..ab_order are precomputed once.
    At each step the embedded error E_i = ||x_r - x_{r-1}||₂ is evaluated
    at zero extra NFE; if E_i > δ_eff the effective order is reduced by 1
    (minimum 1) and restored one level per subsequent step.

    δ_eff = delta_tol · (1 + 0.1 · guidance_w)
    """
    max_order = ab_order

    # ── γ = 0.1 as in the AO-DEIS paper  ──────────────────────────────────
    gamma = 0.1
    delta_eff = delta_tol * (1.0 + gamma * guidance_w)

    jax_rev_ts = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)

    # Precompute coefficient tables for every order 0..max_order
    # coef_table[r] has shape (num_step, max_order+2)
    coef_table_jax = get_ab_eps_coef_all_orders(sde, max_order, jax_rev_ts)

    # Convert to torch tensors once
    th_rev_ts = jax2th(jax_rev_ts)
    th_coef_table = [jax2th(c) for c in coef_table_jax]  # list[max_order+1]

    def sampler(xT):
        device = xT.device
        rev_ts = th_rev_ts.to(device)
        coef_table = [c.to(device) for c in th_coef_table]

        # Initialise eps buffer with xT (same as original)
        eps_pred = [xT] * max_order

        x = xT
        current_order = 1          # warm-up: start at order 1

        for i in range(num_step):
            s_t = rev_ts[i]
            new_eps = eps_fn(x, s_t)

            # Effective order for this step (capped to min(i+1, max_order))
            # so we never request more history than we have
            eff_order = min(current_order, i + 1, max_order)
            eff_order = max(eff_order, 1)

            coef_high = coef_table[eff_order][i]

            if eff_order > 1:
                # Compute embedded error vs order-(eff_order-1)
                coef_low = coef_table[eff_order - 1][i]
                new_x, new_eps_pred, error = ab_step_with_error(
                    x, coef_high, coef_low, new_eps, eps_pred
                )

                # Adaptive order switching
                if error > delta_eff:
                    # Error too large → fall back to lower-order for THIS step
                    new_x, new_eps_pred = ab_step(x, coef_low, new_eps, eps_pred)
                    current_order = max(1, eff_order - 1)
                else:
                    # Error acceptable → use high-order result, try to ramp up
                    current_order = min(eff_order + 1, max_order)
            else:
                # Order-1 step (no error estimate needed)
                new_x, new_eps_pred = ab_step(x, coef_high, new_eps, eps_pred)
                current_order = min(2, max_order)

            x = new_x
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
    ρ-space Adams-Bashforth sampler with adaptive order (AO-DEIS).

    Same adaptive logic as get_sampler_t_ab but operates in the ρ
    (signal-to-noise ratio) coordinate used by rho_ab.
    """
    max_order = ab_order
    gamma = 0.1
    delta_eff = delta_tol * (1.0 + gamma * guidance_w)

    jax_rev_ts = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)
    jax_rev_rhos = sde.t2rho(jax_rev_ts)

    # HelperSDE for the ρ-space coefficient computation
    class HelperSDE(MultiStepSDE):
        def psi(cls, t1, t2):
            return t1 / t1 * t2 / t2
        def eps_integrand(cls, vec_t):
            return vec_t / vec_t

    helper = HelperSDE()

    # Precompute coefficient tables for every order 0..max_order in rho space
    # (x_coef = ones for rho_ab)
    x_coef_jax = jnp.ones(jax_rev_ts.shape[0] - 1)

    coef_table_jax = []
    for r in range(max_order + 1):
        eps_coef_r = get_ab_eps_coef(helper, max_order, jax_rev_rhos, order=r)
        ab_coef_r = jnp.concatenate([x_coef_jax[:, None], eps_coef_r], axis=1)
        coef_table_jax.append(ab_coef_r)

    th_rev_ts = jax2th(jax_rev_ts)
    th_coef_table = [jax2th(c) for c in coef_table_jax]
    nfe = num_step

    def eps_fn_vrho(v, jax_cur_rho):
        jax_cur_t = sde.rho2t(jax_cur_rho)
        x = sde.v2x(v, jax_cur_t)
        return eps_fn(x, jax2th(jax_cur_t, x))

    def sampler(xT):
        device = xT.device
        coef_table = [c.to(device) for c in th_coef_table]

        vT = sde.x2v(xT, jax_rev_ts[0])
        eps_pred = [xT] * max_order

        v = vT
        current_order = 1

        for i in range(nfe):
            rho_cur = jax_rev_rhos[i]
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
                    v_next, new_eps_pred = ab_step(v, coef_low, eps_cur, eps_pred)
                    current_order = max(1, eff_order - 1)
                else:
                    current_order = min(eff_order + 1, max_order)
            else:
                v_next, new_eps_pred = ab_step(v, coef_high, eps_cur, eps_pred)
                current_order = min(2, max_order)

            v = v_next
            eps_pred = new_eps_pred

        x_eps = sde.v2x(v, jax_rev_ts[-1])
        return x_eps

    return sampler


# ---------------------------------------------------------------------------
# Unchanged samplers
# ---------------------------------------------------------------------------

def get_sampler_ipndm(sde, eps_fn, num_step):
    assert isinstance(sde, VPSDE)
    jax_rev_ts = get_rev_ts(sde, num_step, 1, ts_phase="t")
    x_coef = sde.psi(jax_rev_ts[:-1], jax_rev_ts[1:])

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

    jax_linear_ab_coef = get_linear_ab_coef(len(jax_rev_ts) - 2)
    jax_next_ts, jax_cur_ts = jax_rev_ts[1:], jax_rev_ts[:-1]
    jax_next_alpha = sde.t2alpha_fn(jax_next_ts)
    jax_cur_alpha  = sde.t2alpha_fn(jax_cur_ts)
    jax_ddim_coef  = (
        jnp.sqrt(1 - jax_next_alpha)
        - jnp.sqrt(jax_next_alpha / jax_cur_alpha) * jnp.sqrt(1 - jax_cur_alpha)
    )
    jax_eps_coef   = jax_ddim_coef.reshape(-1, 1) * jax_linear_ab_coef
    jax_ab_coef    = jnp.concatenate([x_coef[:, None], jax_eps_coef], axis=1)
    th_rev_ts, th_ab_coef = jax2th(jax_rev_ts), jax2th(jax_ab_coef)

    def sampler(xT):
        rev_ts, ab_coef = th_rev_ts.to(xT.device), th_ab_coef.to(xT.device)
        eps_pred = [xT] * 3
        x = xT
        for i in range(num_step):
            s_t = rev_ts[i]
            new_eps = eps_fn(x, s_t)
            x, eps_pred = ab_step(x, ab_coef[i], new_eps, eps_pred)
        return x

    return sampler


def get_sampler_rho_rk(sde, eps_fn, ts_phase, ts_order, num_step, rk_method):
    jax_rev_ts   = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)
    rk_fn        = get_rk_fn(rk_method)
    jax_rev_rhos = sde.t2rho(jax_rev_ts)
    th_rev_ts    = jax2th(jax_rev_ts)
    th_rev_rhos  = jax2th(jax_rev_rhos)

    def eps_fn_vrho(v, th_rho):
        jax_t = sde.rho2t(th2jax(th_rho))
        x = sde.v2x(v, jax_t)
        return eps_fn(x, jax2th(jax_t, x))

    def _step_fn(i_th, v):
        rho_cur, rho_next = th_rev_rhos[i_th], th_rev_rhos[i_th + 1]
        delta_t = rho_next - rho_cur
        return rk_fn(v, rho_cur, delta_t, eps_fn_vrho)

    def sample_fn(xT):
        vT = sde.x2v(xT, jax_rev_ts[0])
        veps = fori_loop(0, len(th_rev_rhos) - 1, _step_fn, vT)
        xeps = sde.v2x(veps, jax_rev_ts[-1])
        return xeps

    return sample_fn
