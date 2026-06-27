"""CPU-only unit tests for the profiling harness (no model/openpi deps).

These prove the plumbing: that we measure a finite device step time, read XLA's FLOP/byte counts, and compute a
sane MFU + roofline bound on a toy jitted matmul. Absolute MFU on CPU is not meaningful (see PLAN.md G6); we use
a synthetic peak just to exercise the arithmetic.
"""

import jax
import jax.numpy as jnp

from openpi.training import profiling


def _toy_compiled_and_step(n: int = 256):
    x = jnp.ones((n, n), jnp.float32)
    w = jnp.ones((n, n), jnp.float32)
    f = jax.jit(lambda a, b: jnp.tanh(a @ b))
    compiled = f.lower(x, w).compile()
    return compiled, (lambda: f(x, w))


def test_step_cost_reports_positive_flops_and_bytes():
    compiled, _ = _toy_compiled_and_step()
    cost = profiling.step_cost(compiled)
    assert cost.flops > 0
    assert cost.bytes_accessed > 0
    # n x n matmul ~ 2*n^3 FLOPs; just assert the right order of magnitude (>= n^3).
    assert cost.flops >= 256**3
    assert cost.arithmetic_intensity > 0


def test_measure_step_time_is_finite_and_positive():
    _, run_step = _toy_compiled_and_step()
    st = profiling.measure_step_time(run_step, warmup=2, iters=8)
    assert st.n == 8
    assert st.median_s > 0
    assert st.p10_s <= st.median_s <= st.p90_s


def test_mfu_report_with_synthetic_peak():
    compiled, run_step = _toy_compiled_and_step()
    st = profiling.measure_step_time(run_step, warmup=2, iters=5)
    cost = profiling.step_cost(compiled)
    # Synthetic peak well above achievable so MFU lands in (0, 1].
    rep = profiling.mfu_report(st, cost, peak_flops=1e15, peak_bandwidth=1e12)
    assert rep.achieved_flops_per_s > 0
    assert rep.mfu is not None and 0.0 < rep.mfu <= 1.0
    assert rep.bound in ("compute", "memory")
    assert "MFU=" in rep.one_line()


def test_mfu_is_none_without_peak():
    compiled, run_step = _toy_compiled_and_step()
    st = profiling.measure_step_time(run_step, warmup=1, iters=3)
    rep = profiling.mfu_report(st, profiling.step_cost(compiled))
    assert rep.mfu is None
    assert rep.bound is None
    assert "MFU=n/a" in rep.one_line()


def test_summarize_step_times_orders_percentiles():
    st = profiling.summarize_step_times([0.030, 0.010, 0.020, 0.040, 0.050])
    assert st.n == 5
    assert st.median_s == 0.030
    assert st.p10_s <= st.median_s <= st.p90_s


def test_profile_config_disabled_by_default():
    assert profiling.ProfileConfig.from_env({}) is None
    assert profiling.ProfileConfig.from_env({"SHARDER_PROFILE": "0"}) is None


def test_profile_config_from_env_parses_window_and_peaks():
    cfg = profiling.ProfileConfig.from_env(
        {
            "SHARDER_PROFILE": "1",
            "SHARDER_PROFILE_START": "5",
            "SHARDER_PROFILE_STEPS": "3",
            "SHARDER_PEAK_FLOPS": "312e12",
        }
    )
    assert cfg is not None
    assert (cfg.start, cfg.steps) == (5, 3)
    assert cfg.peak_flops == 312e12 and cfg.peak_bandwidth is None
    # window = steps 5,6,7
    assert not cfg.in_window(4)
    assert cfg.in_window(5) and cfg.in_window(7)
    assert not cfg.in_window(8)
    assert cfg.is_last(7) and not cfg.is_last(6)
