"""Device-side profiling: step time, FLOPs/bytes, MFU, and roofline position.

This is a small, self-contained harness used to *measure* the training step so that later changes (multi-host,
sharding) can be evaluated, not guessed. It deliberately has **no dependency on the model or openpi**: it
operates on (a) a zero-arg ``run_step`` thunk that launches one device step, and (b) the *compiled* step's XLA
``cost_analysis()``. That keeps it unit-testable on CPU with a toy jitted function.

Mechanics / why this is "device-side", not wall-clock guesswork:
  - Step time is measured by launching the step and ``jax.block_until_ready`` on its outputs, so we time actual
    device execution (JAX dispatch is async; without the block we'd time Python dispatch only). We report the
    **median** over a window to reject dispatch/host jitter.
  - FLOPs and bytes come from XLA's own ``compiled.cost_analysis()`` (the keys ``"flops"`` and
    ``"bytes accessed"`` on jax 0.5.3) — i.e. the compiler's count for the exact executable, not a hand model.
  - MFU = achieved_FLOP/s ÷ peak_device_FLOP/s. Roofline position compares arithmetic intensity (FLOP/byte) to
    the device ridge point (peak_FLOP/s ÷ peak_bandwidth).

Absolute MFU is only meaningful with a real ``peak_flops`` for the accelerator in use; on simulated CPU devices
it validates the plumbing and gives relative step time only. See PLAN.md (G6).
"""

from __future__ import annotations

import dataclasses
import statistics
import time
from collections.abc import Callable
from typing import Any

import jax

# Approximate dense bf16 peaks (FLOP/s) for convenience only. VERIFY per hardware before trusting MFU — these
# are nominal spec numbers and ignore sparsity/clock variation. None of V1's correctness depends on them.
PEAK_BF16_FLOPS: dict[str, float] = {
    "a100": 312e12,
    "h100": 989e12,  # SXM, dense bf16
    "tpu-v4": 275e12,
    "tpu-v5e": 197e12,
    "tpu-v5p": 459e12,
}


@dataclasses.dataclass(frozen=True)
class StepTime:
    """Per-step wall time measured with device synchronization, in seconds."""

    median_s: float
    mean_s: float
    p10_s: float
    p90_s: float
    n: int


@dataclasses.dataclass(frozen=True)
class StepCost:
    """XLA-reported cost for one compiled step."""

    flops: float
    bytes_accessed: float

    @property
    def arithmetic_intensity(self) -> float:
        """FLOP per byte accessed — the x-axis of the roofline."""
        return self.flops / self.bytes_accessed if self.bytes_accessed else float("inf")


@dataclasses.dataclass(frozen=True)
class MFUReport:
    step_time: StepTime
    cost: StepCost
    achieved_flops_per_s: float
    mfu: float | None  # None when peak_flops is unknown
    bound: str | None  # "compute" | "memory" | None when peak_bw unknown

    def one_line(self) -> str:
        ms = self.step_time.median_s * 1e3
        tflops = self.achieved_flops_per_s / 1e12
        mfu = "n/a" if self.mfu is None else f"{self.mfu * 100:.1f}%"
        bound = self.bound or "n/a"
        ai = self.cost.arithmetic_intensity
        return (
            f"step={ms:.2f}ms  achieved={tflops:.2f} TFLOP/s  MFU={mfu}  "
            f"AI={ai:.1f} FLOP/B  bound={bound}  (median of {self.step_time.n})"
        )


def measure_step_time(run_step: Callable[[], Any], *, warmup: int = 3, iters: int = 20) -> StepTime:
    """Time ``run_step`` over ``iters`` measured launches after ``warmup`` launches.

    ``run_step`` must launch exactly one step and return its output pytree; we ``block_until_ready`` on it so we
    measure device execution rather than async dispatch. The first calls also absorb JIT compilation.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    for _ in range(max(0, warmup)):
        jax.block_until_ready(run_step())

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(run_step())
        samples.append(time.perf_counter() - t0)

    samples.sort()
    return StepTime(
        median_s=statistics.median(samples),
        mean_s=statistics.fmean(samples),
        p10_s=samples[int(0.1 * (len(samples) - 1))],
        p90_s=samples[int(0.9 * (len(samples) - 1))],
        n=len(samples),
    )


def step_cost(compiled: Any) -> StepCost:
    """Extract FLOPs and bytes from a compiled executable's ``cost_analysis()``.

    ``cost_analysis`` returns a dict on jax 0.5.3 (or, for some backends, a list of per-executable dicts — we sum
    those). Missing keys default to 0.0 so this never raises on backends that don't populate them.
    """
    ca = compiled.cost_analysis()
    if isinstance(ca, list):  # some backends return one dict per partition/executable
        flops = sum(float(d.get("flops", 0.0)) for d in ca)
        bytes_accessed = sum(float(d.get("bytes accessed", 0.0)) for d in ca)
    else:
        flops = float(ca.get("flops", 0.0))
        bytes_accessed = float(ca.get("bytes accessed", 0.0))
    return StepCost(flops=flops, bytes_accessed=bytes_accessed)


def mfu_report(
    step_time: StepTime,
    cost: StepCost,
    *,
    peak_flops: float | None = None,
    peak_bandwidth: float | None = None,
) -> MFUReport:
    """Combine measured time + compiler cost into achieved FLOP/s, MFU, and roofline bound."""
    achieved = cost.flops / step_time.median_s if step_time.median_s > 0 else 0.0
    mfu = (achieved / peak_flops) if peak_flops else None
    bound: str | None = None
    if peak_flops and peak_bandwidth:
        ridge = peak_flops / peak_bandwidth  # FLOP/byte where the device transitions compute<->memory bound
        bound = "compute" if cost.arithmetic_intensity >= ridge else "memory"
    return MFUReport(step_time=step_time, cost=cost, achieved_flops_per_s=achieved, mfu=mfu, bound=bound)


def profile_step(
    run_step: Callable[[], Any],
    compiled: Any,
    *,
    peak_flops: float | None = None,
    peak_bandwidth: float | None = None,
    warmup: int = 3,
    iters: int = 20,
    trace_dir: str | None = None,
) -> MFUReport:
    """End-to-end: optionally capture a ``jax.profiler`` trace while timing the step, then build the report."""
    if trace_dir is not None:
        with jax.profiler.trace(trace_dir):
            st = measure_step_time(run_step, warmup=warmup, iters=iters)
    else:
        st = measure_step_time(run_step, warmup=warmup, iters=iters)
    return mfu_report(st, step_cost(compiled), peak_flops=peak_flops, peak_bandwidth=peak_bandwidth)
