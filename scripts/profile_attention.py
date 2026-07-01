"""Honest evaluation of efficient attention for gemma_2b — time, CORRECTNESS, memory, across seq lengths.

The step-breakdown reports attention ~0% because XLA fuses gemma's masked softmax into the neighbouring einsum
fusion on GPU. This isolates one attention block at real gemma_2b shapes (8q/1kv GQA, head_dim=256) and compares:

  naive       : exactly gemma.py — einsum(QK, fp32) -> masked softmax -> einsum(·V)  (materializes T×S matrix)
  flash-xla   : jax.nn.dot_product_attention(..., implementation="xla")
  flash-cudnn : jax.nn.dot_product_attention(..., implementation="cudnn")  (H100 flash kernel; may reject head_dim)

For each seq we report median latency, the max abs diff vs naive (CORRECTNESS), and the attention-matrix memory
naive must hold (which flash avoids). Sweeping seq shows where flash matters (attention cost grows ~seq²).

Note: PagedAttention is an inference-serving KV-cache technique (autoregressive decode) — it does NOT apply to
this training step (no KV cache). It would only matter for pi0-FAST *serving*.
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np


def _median_ms(fn, warmup=5, iters=30):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    s = []
    for _ in range(iters):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        s.append((time.perf_counter() - t) * 1e3)
    return statistics.median(s)


def _build(B, T, Nq, Nkv, Hd, scale):
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((B, T, Nkv, Nq // Nkv, Hd), dtype=np.float32)).astype(jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((B, T, Nkv, Hd), dtype=np.float32)).astype(jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((B, T, Nkv, Hd), dtype=np.float32)).astype(jnp.bfloat16)
    mask = jnp.tril(jnp.ones((T, T), bool))[None, None]

    @jax.jit
    def naive(q, k, v):
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32) * scale
        logits = jnp.where(mask[:, :, None], logits, jnp.finfo(jnp.float32).min)
        probs = jax.nn.softmax(logits, axis=-1).astype(jnp.bfloat16)
        return jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)

    qf = q.reshape(B, T, Nq, Hd)
    kf, vf = k.reshape(B, T, Nkv, Hd), v.reshape(B, T, Nkv, Hd)

    def flash(impl):
        @jax.jit
        def f(qf, kf, vf):
            out = jax.nn.dot_product_attention(qf, kf, vf, is_causal=True, scale=scale, implementation=impl)
            return out.reshape(B, T, Nkv, Nq // Nkv, Hd)  # match naive layout for the correctness check

        return f

    return (naive, (q, k, v)), flash, (qf, kf, vf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqs", type=int, nargs="+", default=[866, 2048, 4096])
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--step-ms", type=float, default=211.9)
    p.add_argument("--head-dim", type=int, default=256)
    args = p.parse_args()

    B, Nq, Nkv, Hd = args.batch, 8, 1, args.head_dim
    scale = 1.0 / np.sqrt(Hd)
    print(f"gemma_2b attention | B={B} heads={Nq}q/{Nkv}kv Hd={Hd} depth={args.depth} step={args.step_ms}ms\n")

    for T in args.seqs:
        (naive, nargs), flash, fargs = _build(B, T, Nq, Nkv, Hd, scale)
        ref = jax.block_until_ready(naive(*nargs))
        naive_ms = _median_ms(lambda: naive(*nargs))
        # attention matrix naive must hold: logits fp32 [B,Nq,T,T] + probs bf16.
        attn_mem_gb = B * Nq * T * T * (4 + 2) / 1e9
        print(f"seq={T}:  naive {naive_ms:.3f} ms/layer | {naive_ms*args.depth:.1f} ms/step "
              f"({naive_ms*args.depth/args.step_ms*100:.1f}%) | attn-matrix {attn_mem_gb:.2f} GB/layer")
        for impl in ("xla", "cudnn"):
            try:
                f = flash(impl)
                out = jax.block_until_ready(f(*fargs))
                diff = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
                ms = _median_ms(lambda: f(*fargs))
                print(f"         flash-{impl:5} {ms:.3f} ms/layer | {ms*args.depth:.1f} ms/step "
                      f"({ms*args.depth/args.step_ms*100:.1f}%) | maxdiff-vs-naive {diff:.3f} | attn-matrix ~0 GB")
            except Exception as e:  # noqa: BLE001
                print(f"         flash-{impl:5} UNAVAILABLE: {type(e).__name__}: {str(e)[:90]}")
        print()


if __name__ == "__main__":
    main()
