"""
AO-DEIS: Adaptive-Order Exponential Integrators for Fast and Guided Diffusion Sampling
=========================================================================================
Authors: Muhammad Abdul Aleem, Noman Ali, Muhammad Salman Aslam, Zain Anjum, Muhammad Waleed
Course:  Deep Learning Spring 2026 – Information Technology University, Lahore, Pakistan

This module implements AO-DEIS on top of the official DEIS codebase:
  https://github.com/qsh-zh/deis

Key contributions over fixed-order DEIS:
1. Zero-cost smoothness indicator  delta_i  computed from the existing epsilon-buffer
2. Adaptive order selection  r(t_i) based on delta_i vs tau_smooth
3. Guidance-aware threshold  delta_eff = delta_0 * (1 + gamma * w)  for CFG scale w
4. Embedded error estimator (EEE) for automatic threshold calibration
"""

import torch
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Smoothness indicator
# ---------------------------------------------------------------------------

def compute_smoothness_indicator(
    eps_new: torch.Tensor,
    eps_prev: torch.Tensor,
    eps_stab: float = 1e-6,
) -> float:
    """
    Compute the zero-cost smoothness indicator delta_i (Eq. 3 in the proposal).

        delta_i = ||eps_new - eps_prev||_2 / (||eps_prev||_2 + eps_stab)

    This re-uses score evaluations already in the epsilon-buffer, so it costs
    *zero* additional network function evaluations (NFE).

    Parameters
    ----------
    eps_new  : current score evaluation  eps_theta(x_t, t)
    eps_prev : previous score evaluation eps_theta(x_{t+1}, t+1)
    eps_stab : small constant to prevent division by zero (default 1e-6)

    Returns
    -------
    delta : float  – relative variation; large → non-smooth (use low order);
                     small → smooth (use high order)
    """
    diff  = (eps_new - eps_prev).norm().item()
    denom = eps_prev.norm().item() + eps_stab
    return diff / denom


# ---------------------------------------------------------------------------
# 2.  Guidance-aware effective threshold
# ---------------------------------------------------------------------------

def get_effective_threshold(
    tau_smooth: float,
    cfg_scale: float = 1.0,
    gamma: float = 0.1,
) -> float:
    """
    Scale tau_smooth by the CFG guidance scale w (Eq. 5 in the proposal).

        delta_eff = tau_smooth * (1 + gamma * w)

    Higher CFG scale amplifies ODE stiffness, so we *tighten* the smoothness
    criterion (require lower delta to use high order), which automatically
    reduces r during high-guidance steps.

    Parameters
    ----------
    tau_smooth : base smoothness threshold (hyperparameter, tuned on val set)
    cfg_scale  : classifier-free guidance scale w  (1.0 = unconditional)
    gamma      : sensitivity to guidance scale (default 0.1)

    Returns
    -------
    delta_eff : float
    """
    return tau_smooth * (1.0 + gamma * cfg_scale)


# ---------------------------------------------------------------------------
# 3.  Adaptive order selector
# ---------------------------------------------------------------------------

def select_order(
    delta_i: float,
    delta_eff: float,
    order_low: int = 1,
    order_high: int = 3,
) -> int:
    """
    Choose polynomial extrapolation order at step i (Eq. 4, binary variant).

        r(t_i) = order_low  if delta_i > delta_eff   (non-smooth regime)
               = order_high if delta_i <= delta_eff  (smooth regime)

    Kept for backward compatibility / ablation ("binary AO-DEIS"). The
    default sampler now uses `select_order_smooth` below, which avoids the
    abrupt order-1 -> order-3 jump.
    """
    return order_low if delta_i > delta_eff else order_high


def select_order_smooth(
    delta_i: float,
    delta_eff: float,
    order_low: int = 1,
    order_mid: int = 2,
    order_high: int = 3,
    mid_ratio: float = 0.5,
) -> int:
    """
    Graded order selection across {order_low, order_mid, order_high}
    (Eq. 4, smooth variant — addresses the jerky binary-switch limitation).

    Two thresholds are derived from delta_eff instead of one:
        delta_eff_high = delta_eff             (boundary for order_mid)
        delta_eff_low  = delta_eff * mid_ratio (tighter boundary for order_high)

        r(t_i) = order_low   if delta_i  > delta_eff_high
               = order_mid   if delta_eff_low < delta_i <= delta_eff_high
               = order_high  if delta_i <= delta_eff_low

    This removes the single abrupt 1->3 jump: the trajectory now passes
    through order 2 whenever delta_i sits in an intermediate band, so the
    step-size/polynomial-degree change between consecutive steps is at most
    one order instead of two.
    """
    delta_eff_low = delta_eff * mid_ratio
    if delta_i > delta_eff:
        return order_low
    elif delta_i > delta_eff_low:
        return order_mid
    else:
        return order_high


# ---------------------------------------------------------------------------
# 4.  Embedded Error Estimator (EEE)
# ---------------------------------------------------------------------------

def embedded_error_estimator(
    x_high: torch.Tensor,
    x_low: torch.Tensor,
) -> float:
    """
    Compute the embedded error estimate E_i = ||x_high - x_low||_2.

    Both x_high (order r) and x_low (order r-1) are computed from the SAME
    epsilon-buffer pass, so no extra NFE is required.  When E_i > delta_tol,
    the order is reduced.

    Parameters
    ----------
    x_high : denoised sample at order r
    x_low  : denoised sample at order r-1

    Returns
    -------
    E_i : float
    """
    return (x_high - x_low).norm().item()


# ---------------------------------------------------------------------------
# 5.  AO-DEIS sampler  (wraps tAB-DEIS with adaptive order)
# ---------------------------------------------------------------------------

class AODEISSampler:
    """
    Adaptive-Order DEIS sampler.

    Usage
    -----
    sampler = AODEISSampler(sde, eps_fn, tau_smooth=0.1, cfg_scale=1.0)
    x0 = sampler.sample(xT, num_steps=10)
    """

    def __init__(
        self,
        sde,
        eps_fn,
        tau_smooth: float = 0.1,
        cfg_scale: float = 1.0,
        gamma: float = 0.1,
        order_low: int = 1,
        order_high: int = 3,
        delta_tol: Optional[float] = None,
        eps_stab: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        sde          : VPSDE instance from th_deis
        eps_fn       : callable(x, t) → score evaluation eps_theta(x,t)
        tau_smooth   : base smoothness threshold
        cfg_scale    : CFG guidance scale w (1.0 = unconditional)
        gamma        : guidance sensitivity coefficient
        order_low    : order used in non-smooth (high-noise) regime
        order_high   : order used in smooth (low-noise) regime
        delta_tol    : tolerance for EEE-based order reduction (None = disabled)
        eps_stab     : stabiliser for delta_i denominator
        verbose      : log per-step order decisions
        """
        self.sde        = sde
        self.eps_fn     = eps_fn
        self.tau_smooth = tau_smooth
        self.cfg_scale  = cfg_scale
        self.gamma      = gamma
        self.order_low  = order_low
        self.order_high = order_high
        self.delta_tol  = delta_tol
        self.eps_stab   = eps_stab
        self.verbose    = verbose

        # Tracking for analysis
        self.order_log: List[int]   = []
        self.delta_log: List[float] = []
        self.error_log: List[float] = []

    # ------------------------------------------------------------------
    def _get_timesteps(self, num_steps: int, ts_order: int = 2) -> torch.Tensor:
        """Quadratic timestep schedule (Song et al., 2020)."""
        import jax.numpy as jnp
        from .sde import get_rev_ts
        from .helper import jax2th

        rev_ts = get_rev_ts(self.sde, num_steps, ts_order, ts_phase="t")
        return jax2th(rev_ts)

    # ------------------------------------------------------------------
    def _compute_ab_coef(self, rev_ts: torch.Tensor, order: int):
        """Pre-compute tAB-DEIS coefficients for a given order."""
        import jax.numpy as jnp
        from .multistep import get_ab_eps_coef
        from .helper import jax2th, th2jax

        jax_rev_ts = th2jax(rev_ts)
        x_coef  = self.sde.psi(jax_rev_ts[:-1], jax_rev_ts[1:])
        eps_coef = get_ab_eps_coef(self.sde, order, jax_rev_ts, order)
        ab_coef  = jnp.concatenate([x_coef[:, None], eps_coef], axis=1)
        return jax2th(ab_coef)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        xT: torch.Tensor,
        num_steps: int = 10,
        ts_order: int = 2,
    ) -> torch.Tensor:
        """
        Run AO-DEIS sampling.

        Parameters
        ----------
        xT        : initial noise tensor, shape (B, C, H, W)
        num_steps : number of denoising steps (NFE = num_steps)
        ts_order  : timestep schedule power (2 = quadratic, recommended)

        Returns
        -------
        x0 : denoised sample, same shape as xT
        """
        self.order_log.clear()
        self.delta_log.clear()
        self.error_log.clear()

        device = xT.device

        # Pre-compute timesteps and coefficients for both orders
        rev_ts = self._get_timesteps(num_steps, ts_order).to(device)

        # Pre-compute AB coefficients for every order needed: order_low,
        # order_high, and everything in between (the smooth selector can
        # land on an intermediate order, and bootstrap can land on any
        # order from 1 up to order_high).
        needed_orders = sorted(set(range(1, self.order_high + 1)) | {self.order_low, self.order_high})
        ab_coef_by_order = {
            r_: self._compute_ab_coef(rev_ts, r_).to(device)
            for r_ in needed_orders
        }
        ab_coef_high = ab_coef_by_order[self.order_high]
        ab_coef_low  = ab_coef_by_order[self.order_low]

        delta_eff = get_effective_threshold(self.tau_smooth, self.cfg_scale, self.gamma)

        # ------------------------------------------------------------------
        # Bootstrap-safe epsilon buffer.
        #
        # Previously this buffer was pre-filled with clones of xT, which is
        # not a real score evaluation -- comparing eps_new against xT at
        # step 0 produces a meaningless, near-random delta_0 and can corrupt
        # the very first multistep update. A proper multistep solver instead
        # *bootstraps*: it starts at order 1 (no history needed) and only
        # raises the order once enough genuine eps_theta evaluations have
        # accumulated in the buffer.
        #
        # `eps_buf` therefore starts EMPTY and is grown with real
        # evaluations only. `n_real` tracks how many genuine evaluations
        # have been produced so far.
        # ------------------------------------------------------------------
        buf_size = max(self.order_high, self.order_low) + 1
        eps_buf: List[torch.Tensor] = []
        n_real = 0

        x = xT.clone()

        for i in range(num_steps):
            t_cur = rev_ts[i]

            # ---- score evaluation ----------------------------------------
            eps_new = self.eps_fn(x, t_cur)

            # ---- bootstrap cap: how high can the order legally go? -------
            # n_real previous real evaluations are available *before* this
            # step's update is applied, so the highest order an Adams-type
            # multistep method can safely use right now is n_real + 1.
            max_safe_order = min(self.order_high, n_real + 1)

            if n_real == 0:
                # No real history yet (step 0): smoothness is undefined,
                # not chaotic. Force the lowest safe order and skip the
                # indicator rather than computing a meaningless delta
                # against noise.
                delta_i = None
                r = min(self.order_low, max_safe_order)
            else:
                eps_prev = eps_buf[0]
                delta_i  = compute_smoothness_indicator(eps_new, eps_prev, self.eps_stab)
                r = select_order_smooth(
                    delta_i, delta_eff,
                    order_low=self.order_low,
                    order_mid=max((self.order_low + self.order_high) // 2, self.order_low),
                    order_high=self.order_high,
                )
                r = min(r, max_safe_order)  # never exceed what history allows

            self.delta_log.append(delta_i if delta_i is not None else float("nan"))

            # ---- coefficient lookup for the chosen / bootstrap-capped order
            coef_for_r = ab_coef_by_order[r][i]

            # Apply EEE if enabled and we're actually attempting high order
            if self.delta_tol is not None and r == self.order_high:
                coef_h = ab_coef_high[i].to(device)
                coef_l = ab_coef_low[i].to(device)
                x_high = self._ab_update(x, coef_h, eps_new, eps_buf)
                x_low  = self._ab_update(x, coef_l, eps_new, eps_buf)
                E_i    = embedded_error_estimator(x_high, x_low)
                self.error_log.append(E_i)
                if E_i > self.delta_tol:
                    r = self.order_low
                    x_new = x_low
                else:
                    x_new = x_high
            else:
                x_new = self._ab_update(x, coef_for_r.to(device), eps_new, eps_buf)

            self.order_log.append(r)

            # ---- grow the epsilon buffer with the REAL evaluation ---------
            eps_buf.insert(0, eps_new)
            if len(eps_buf) > buf_size:
                eps_buf.pop()
            n_real += 1

            x = x_new

            if self.verbose:
                d_str = f"{delta_i:.4f}" if delta_i is not None else "  n/a"
                logger.info(f"Step {i:3d} | t={t_cur:.4f} | delta={d_str} | r={r}")

        return x

    # ------------------------------------------------------------------
    @staticmethod
    def _ab_update(
        x: torch.Tensor,
        coef: torch.Tensor,
        eps_new: torch.Tensor,
        eps_buf: List[torch.Tensor],
    ) -> torch.Tensor:
        """Apply one Adams-Bashforth tAB-DEIS update step."""
        x_coef  = coef[0]
        eps_coef = coef[1:]

        full_eps = [eps_new] + list(eps_buf)
        out = x_coef * x
        for c, e in zip(eps_coef, full_eps):
            out = out + c * e
        return out

    # ------------------------------------------------------------------
    def get_order_stats(self) -> dict:
        """Return per-step order and delta statistics for analysis."""
        orders = np.array(self.order_log)
        deltas = np.array(self.delta_log)
        stats  = {
            "mean_order"   : float(orders.mean()) if len(orders) > 0 else 0.0,
            "frac_high"    : float((orders == self.order_high).mean()) if len(orders) > 0 else 0.0,
            # nanmean: delta_log[0] is NaN during bootstrap (no real history
            # yet, so smoothness is undefined rather than computed from a
            # placeholder noise vector).
            "mean_delta"   : float(np.nanmean(deltas)) if len(deltas) > 0 else 0.0,
            "orders"       : orders.tolist(),
            "deltas"       : deltas.tolist(),
        }
        return stats
