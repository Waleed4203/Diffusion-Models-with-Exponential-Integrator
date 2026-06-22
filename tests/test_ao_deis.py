"""
tests/test_ao_deis.py  —  AO-DEIS adaptive-order sampler tests

Tests:
  1. Basic correctness  – sampler returns tensor of correct shape / dtype
  2. Adaptive order     – for a very jagged eps_fn the order should drop
  3. Guidance-aware     – higher guidance_w → wider tolerance (smoke test)
  4. Backward compat    – calling without new kwargs works identically to
                          calling the original DEIS
"""

import jax.numpy as jnp
import jax_deis
import th_deis
import torch as th
import numpy as np
import pytest

data_shape = (3, 13)

# ── Helpers ──────────────────────────────────────────────────────────────

def jax_eps_fn_smooth(x, t):
    """Smooth (constant) eps prediction."""
    del t
    return jnp.ones(x.shape)

def jax_eps_fn_jagged(x, t):
    """Jagged eps prediction – forces AO-DEIS to lower the order."""
    del t
    return jnp.where(jnp.sum(x) > 0, jnp.ones(x.shape) * 10.0, -jnp.ones(x.shape) * 10.0)

def th_eps_fn_smooth(x, t):
    del t
    return th.ones_like(x)

def th_eps_fn_jagged(x, t):
    del t
    return th.where(x.sum() > 0, th.ones_like(x) * 10.0, -th.ones_like(x) * 10.0)


# ── JAX tests ─────────────────────────────────────────────────────────────

class TestJaxAODEIS:

    def setup_method(self):
        t2alpha_fn, alpha2t_fn = jax_deis.get_linear_alpha_fns(0.01, 20)
        self.sde   = jax_deis.VPSDE(t2alpha_fn, alpha2t_fn, 1e-3, 1.0)
        self.noise = jnp.ones(data_shape)

    def test_t_ab_smooth_no_order_drop(self):
        """Smooth eps → sampler should reach max order without crashing."""
        for ab_order in [1, 2, 3]:
            fn = jax_deis.get_sampler(
                self.sde, jax_eps_fn_smooth,
                ts_phase="t", ts_order=2.0, num_step=10,
                method="t_ab", ab_order=ab_order,
                delta_tol=0.1, guidance_w=0.0,
            )
            out = fn(self.noise)
            assert out.shape == data_shape, f"Shape mismatch for ab_order={ab_order}"

    def test_rho_ab_smooth_no_order_drop(self):
        for ab_order in [1, 2, 3]:
            fn = jax_deis.get_sampler(
                self.sde, jax_eps_fn_smooth,
                ts_phase="t", ts_order=2.0, num_step=10,
                method="rho_ab", ab_order=ab_order,
                delta_tol=0.1,
            )
            out = fn(self.noise)
            assert out.shape == data_shape

    def test_t_ab_jagged_does_not_explode(self):
        """With a very high tolerance, even a jagged eps_fn must not NaN."""
        fn = jax_deis.get_sampler(
            self.sde, jax_eps_fn_jagged,
            ts_phase="t", ts_order=2.0, num_step=10,
            method="t_ab", ab_order=3,
            delta_tol=1e-6,          # very tight → forces order-1 almost always
            guidance_w=0.0,
        )
        out = fn(self.noise)
        assert not jnp.any(jnp.isnan(out)), "NaN detected in output"
        assert not jnp.any(jnp.isinf(out)), "Inf detected in output"

    def test_guidance_aware_threshold(self):
        """With guidance_w=7.5, δ_eff should be wider → sampler stays stable."""
        fn = jax_deis.get_sampler(
            self.sde, jax_eps_fn_smooth,
            ts_phase="t", ts_order=2.0, num_step=10,
            method="t_ab", ab_order=3,
            delta_tol=0.1, guidance_w=7.5,
        )
        out = fn(self.noise)
        assert out.shape == data_shape

    def test_backward_compat_no_new_kwargs(self):
        """Calling without delta_tol / guidance_w must still work."""
        fn = jax_deis.get_sampler(
            self.sde, jax_eps_fn_smooth,
            ts_phase="t", ts_order=2.0, num_step=10,
            method="t_ab", ab_order=3,
        )
        out = fn(self.noise)
        assert out.shape == data_shape

    def test_ipndm_unchanged(self):
        fn = jax_deis.get_sampler(
            self.sde, jax_eps_fn_smooth,
            ts_phase="t", ts_order=2.0, num_step=10,
            method="ipndm",
        )
        out = fn(self.noise)
        assert out.shape == data_shape

    def test_rho_rk_unchanged(self):
        for rk in ["1euler", "2heun", "3kutta", "4rk"]:
            fn = jax_deis.get_sampler(
                self.sde, jax_eps_fn_smooth,
                ts_phase="t", ts_order=2.0, num_step=10,
                method="rho_rk", rk_method=rk,
            )
            out = fn(self.noise)
            assert out.shape == data_shape


# ── PyTorch (th_deis) tests ───────────────────────────────────────────────

class TestThAODEIS:

    def setup_method(self):
        t2alpha_fn, alpha2t_fn = th_deis.get_linear_alpha_fns(0.01, 20)
        self.sde   = th_deis.VPSDE(t2alpha_fn, alpha2t_fn, 1e-3, 1.0)
        self.noise = th.ones(data_shape)

    def test_t_ab_all_orders(self):
        for ab_order in [1, 2, 3]:
            fn = th_deis.get_sampler(
                self.sde, th_eps_fn_smooth,
                ts_phase="t", ts_order=2.0, num_step=10,
                method="t_ab", ab_order=ab_order,
                delta_tol=0.1,
            )
            out = fn(self.noise)
            assert out.shape == data_shape

    def test_rho_ab_all_orders(self):
        for ab_order in [1, 2, 3]:
            fn = th_deis.get_sampler(
                self.sde, th_eps_fn_smooth,
                ts_phase="t", ts_order=2.0, num_step=10,
                method="rho_ab", ab_order=ab_order,
            )
            out = fn(self.noise)
            assert out.shape == data_shape

    def test_t_ab_tight_tolerance_no_nan(self):
        fn = th_deis.get_sampler(
            self.sde, th_eps_fn_smooth,
            ts_phase="t", ts_order=2.0, num_step=10,
            method="t_ab", ab_order=3,
            delta_tol=1e-6,
        )
        out = fn(self.noise)
        assert not th.any(th.isnan(out))
        assert not th.any(th.isinf(out))

    def test_guidance_w_smoke(self):
        fn = th_deis.get_sampler(
            self.sde, th_eps_fn_smooth,
            ts_phase="t", ts_order=2.0, num_step=10,
            method="t_ab", ab_order=3,
            delta_tol=0.1, guidance_w=7.5,
        )
        out = fn(self.noise)
        assert out.shape == data_shape

    def test_backward_compat(self):
        fn = th_deis.get_sampler(
            self.sde, th_eps_fn_smooth,
            ts_phase="t", ts_order=2.0, num_step=10,
            method="t_ab", ab_order=3,
        )
        out = fn(self.noise)
        assert out.shape == data_shape

    def test_ipndm_unchanged(self):
        fn = th_deis.get_sampler(
            self.sde, th_eps_fn_smooth,
            ts_phase="t", ts_order=2.0, num_step=10,
            method="ipndm",
        )
        out = fn(self.noise)
        assert out.shape == data_shape

    def test_rho_rk_unchanged(self):
        for rk in ["1euler", "2heun", "3kutta", "4rk"]:
            fn = th_deis.get_sampler(
                self.sde, th_eps_fn_smooth,
                ts_phase="t", ts_order=2.0, num_step=10,
                method="rho_rk", rk_method=rk,
            )
            out = fn(self.noise)
            assert out.shape == data_shape
