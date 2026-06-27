"""V2/M1: attribute measured device step time into categories — the DECISION INSTRUMENT.

We never optimize a speculative bottleneck. Before any V2 optimization, this module breaks the step's device
time into:  attention | matmul-FFN | collectives | optimizer | other  (+ data-wait as device idle), so the
breakdown justifies exactly one change.

How it works (cheap-ladder friendly, no TensorFlow/xplane parsing):
  1. From the compiled step's HLO text (`compiled.as_text()`) we map each HLO instruction name -> its
     `metadata op_name`, which carries the full `jax.named_scope` / nnx-module path (e.g.
     ".../attention/exp", ".../mlp/dot_general"). This is how scope survives onto CPU where the Perfetto trace
     itself only records bare instruction names.
  2. We run `jax.profiler.trace` over a window of real steps and parse the emitted Chrome trace
     (`*.trace.json.gz`): each device-op event carries `args.hlo_op` (the instruction name) and a duration.
  3. Join: instruction -> scope/op-type -> category; sum device time per category. data-wait = the fraction of
     wall time the device was idle (wall_window - device_busy).

Limitations (flagged): on CPU, attention/FFN GEMMs are both `dot` ops; we separate *softmax* (the actual
attention headroom per the audit) into `attention` and keep all matmuls in `matmul-FFN`, which matches the V2
decision gate ("don't touch saturated GEMMs; fuse attention softmax"). Optimizer isolation relies on the
optimizer ops being scoped; unscoped elementwise lands in `other`.
"""

from __future__ import annotations

import dataclasses
import glob
import gzip
import json
import re
from collections.abc import Callable
from typing import Any

import jax

CATEGORIES = ("vision", "image-aug", "embedding", "attention", "matmul-FFN", "collectives", "optimizer", "other")

# Patterns built from the REAL pi0 HLO scope vocabulary (dumped from the compiled model), not guesses:
#   SigLIP vision -> encoderblock / MultiHeadDotProductAttention_0 / encoder_norm
#   gemma attention -> attn ; FFN/matmul -> dot_general ; gelu -> erf
#   augmax on-device image augmentation -> pixelwise / map_coordinates / piecewise  (the bulk of old "other")
_COLLECTIVE_RE = re.compile(r"all-reduce|all-gather|reduce-scatter|collective-permute|all-to-all|ppermute")
_MATMUL_RE = re.compile(r"(^|[^a-z])(dot|convolution|conv)([^a-z]|$)|dot_general|dot\.|convolution")
_SOFTMAX_RE = re.compile(r"exponential|logistic|reduce-window|reduce_max|reduce_sum|softmax")
_OPT_RE = re.compile(r"adam|scale_by|optax|optimizer|ema_update|/opt\b")
_IMAGEAUG_RE = re.compile(r"pixelwise|map_coordinates|piecewise|augmax|random_crop|random_resize|image_aug")
_VISION_RE = re.compile(r"encoderblock|siglip|\bvit\b|vision|img_encoder|patch_embed|multiheaddotproductattention")
_EMBED_RE = re.compile(r"embed|embedder|input_embedding|pos_embed|positional|\bwte\b|token_emb")
_FFN_RE = re.compile(r"\bmlp\b|ffn|feed_forward|dense|gelu|swiglu|gating|\berf\b|einsum")


def classify(instruction: str, scope: str) -> str:
    """Map an HLO instruction (bare name) + its scope path (metadata op_name) to a category.

    Priority matters: collectives/optimizer first (unambiguous), then vision/image-aug/embedding (so SigLIP and
    augmentation don't leak into attention/matmul), then gemma attention (softmax only — the audited headroom),
    then matmul-FFN (all GEMMs incl. the attention QK/AV dots, which XLA saturates), else other.
    """
    s = scope.lower()
    name = instruction.lower()

    if _COLLECTIVE_RE.search(name) or _COLLECTIVE_RE.search(s):
        return "collectives"
    if _OPT_RE.search(s):
        return "optimizer"
    # On-device image augmentation (augmax) — surfaced explicitly; was the dominant chunk of old "other".
    if _IMAGEAUG_RE.search(s):
        return "image-aug"
    # SigLIP vision tower (incl. its own MHA softmax — counts as vision, not gemma attention).
    if _VISION_RE.search(s):
        return "vision"
    if _EMBED_RE.search(s):
        return "embedding"
    # Gemma attention: only the softmax ops. The QK^T/AV dots fall through to matmul-FFN below.
    if ("attn" in s or "attention" in s) and (_SOFTMAX_RE.search(name) or _SOFTMAX_RE.search(s)):
        return "attention"
    if _MATMUL_RE.search(name) or _FFN_RE.search(s):
        return "matmul-FFN"
    if _SOFTMAX_RE.search(name):  # bare softmax with no scope
        return "attention"
    return "other"


def _hlo_scope_map(compiled: Any) -> dict[str, str]:
    """instruction-name -> metadata op_name (scope path), parsed from the compiled HLO text."""
    text = compiled.as_text()
    scope_of: dict[str, str] = {}
    inst_re = re.compile(r"%([\w.\-]+)\s*=")
    op_re = re.compile(r'op_name="([^"]*)"')
    for line in text.splitlines():
        inst = inst_re.search(line)
        op = op_re.search(line)
        if inst and op:
            # Trace events use the trailing instruction name (e.g. "dot.38"); key on the bare name.
            scope_of[inst.group(1)] = op.group(1)
    return scope_of


@dataclasses.dataclass(frozen=True)
class Breakdown:
    category_us: dict[str, float]  # device time per category (microseconds, summed over the window)
    device_busy_us: float
    wall_us: float
    n_steps: int

    @property
    def data_wait_us(self) -> float:
        return max(0.0, self.wall_us - self.device_busy_us)

    def percentages(self) -> dict[str, float]:
        """Each category + data-wait as a fraction of WALL time (so they sum to ~1 with data-wait)."""
        total = self.wall_us if self.wall_us > 0 else 1.0
        out = {c: self.category_us.get(c, 0.0) / total for c in CATEGORIES}
        out["data-wait"] = self.data_wait_us / total
        return out

    def dominant(self) -> str:
        return max(self.percentages().items(), key=lambda kv: kv[1])[0]

    def table(self) -> str:
        pct = self.percentages()
        rows = [f"  {'category':<14} {'device ms/step':>14} {'% wall':>8}"]
        for c in (*CATEGORIES, "data-wait"):
            ms = (self.category_us.get(c, self.data_wait_us if c == "data-wait" else 0.0)) / 1e3 / max(1, self.n_steps)
            rows.append(f"  {c:<14} {ms:>14.3f} {pct[c] * 100:>7.1f}%")
        return "\n".join(rows)


def _parse_trace_durations(trace_dir: str) -> dict[str, float]:
    """Sum device-op durations (us) per HLO instruction name from the Chrome trace."""
    files = glob.glob(f"{trace_dir}/**/*.trace.json.gz", recursive=True)
    if not files:
        raise FileNotFoundError(f"no Chrome trace under {trace_dir}")
    with gzip.open(files[0]) as fh:
        events = json.load(fh)["traceEvents"]
    per_inst: dict[str, float] = {}
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        hlo_op = (e.get("args") or {}).get("hlo_op")
        if hlo_op:
            per_inst[hlo_op] = per_inst.get(hlo_op, 0.0) + float(e["dur"])
    return per_inst


def attribute_step(
    run_step: Callable[[], Any],
    compiled: Any,
    *,
    trace_dir: str,
    warmup: int = 3,
    iters: int = 20,
) -> Breakdown:
    """Trace a window of `run_step`, attribute device time to categories using the compiled HLO scopes."""
    import time

    scope_of = _hlo_scope_map(compiled)

    for _ in range(max(0, warmup)):
        jax.block_until_ready(run_step())

    t0 = time.perf_counter()
    with jax.profiler.trace(trace_dir):
        for _ in range(iters):
            jax.block_until_ready(run_step())
    wall_us = (time.perf_counter() - t0) * 1e6

    per_inst = _parse_trace_durations(trace_dir)

    category_us = dict.fromkeys(CATEGORIES, 0.0)
    device_busy_us = 0.0
    for inst, dur in per_inst.items():
        # Trace instruction names may have a ".clone"/suffix; strip to match HLO map best-effort.
        scope = scope_of.get(inst) or scope_of.get(inst.split(".clone")[0]) or ""
        category_us[classify(inst, scope)] += dur
        device_busy_us += dur

    return Breakdown(category_us=category_us, device_busy_us=device_busy_us, wall_us=wall_us, n_steps=iters)
