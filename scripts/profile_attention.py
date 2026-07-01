"""Measure the REAL attention cost at gemma_2b shapes — by ablation, not by the (fused-op) trace.

The step-breakdown reports attention ≈ 0% because on GPU XLA fuses gemma's masked `jax.nn.softmax`
(gemma.py:228) into the surrounding einsum fusion, so the Chrome-trace composition attributes its time to
matmul-FFN. That does NOT mean attention is free. This benchmark isolates one attention block at the real
gemma_2b shapes (8 query / 1 KV head GQA, head_dim 256, seq≈866) and times:

  naive     : exactly gemma.py — einsum(QK, fp32) -> masked softmax -> einsum(·V)
  flash-xla : jax.nn.dot_product_attention(..., implementation="xla")
  flash-cudnn: jax.nn.dot_product_attention(..., implementation="cudnn")   # H100 flash kernel

Per-layer time × depth gives attention's real share of the ~200 ms/step, and naive-vs-flash gives the actual
fused-attention headroom (the V2 gate) — a direct answer, no trace guesswork.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=866)
    p.add_argument("--depth", type=int, default=18)  # gemma_2b layers
    p.add_argument("--step-ms", type=float, default=211.9, help="full gemma_2b step (for %)")
    args = p.parse_args()

    B, T, Nq, Nkv, Hd = args.batch, args.seq, 8, 1, 256
    G = Nq // Nkv  # query groups per kv head (GQA)
    rng = np.random.default_rng(0)
    scale = 1.0 / np.sqrt(Hd)

    # gemma.py layout: q [B,T,Nkv,G,Hd]; k,v [B,S,Nkv,Hd]; block-causal mask [B,1,T,S].
    q = jnp.asarray(rng.standard_normal((B, T, Nkv, G, Hd), dtype=np.float32)).astype(jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((B, T, Nkv, Hd), dtype=np.float32)).astype(jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((B, T, Nkv, Hd), dtype=np.float32)).astype(jnp.bfloat16)
    mask = jnp.tril(jnp.ones((T, T), bool))[None, None]  # causal, [1,1,T,T]

    @jax.jit
    def naive(q, k, v):
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32) * scale
        logits = jnp.where(mask[:, :, None], logits, jnp.finfo(jnp.float32).min)
        probs = jax.nn.softmax(logits, axis=-1).astype(jnp.bfloat16)
        return jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)

    # dot_product_attention wants [B,T,N,H]; GQA (Nq query heads, Nkv kv heads) is supported by broadcasting.
    qf = q.reshape(B, T, Nq, Hd)
    kf, vf = k.reshape(B, T, Nkv, Hd), v.reshape(B, T, Nkv, Hd)

    def _flash(impl):
        @jax.jit
        def f(qf, kf, vf):
            return jax.nn.dot_product_attention(qf, kf, vf, is_causal=True, scale=scale, implementation=impl)

        return f

    naive_ms = _median_ms(lambda: naive(q, k, v))
    results = {"naive": naive_ms}
    for impl in ("xla", "cudnn"):
        try:
            f = _flash(impl)
            results[f"flash-{impl}"] = _median_ms(lambda: f(qf, kf, vf))
        except Exception as e:  # noqa: BLE001
            results[f"flash-{impl}"] = f"unavailable ({type(e).__name__})"

    print(f"\ngemma_2b attention: B={B} seq={T} heads={Nq}q/{Nkv}kv Hd={Hd} depth={args.depth} | step={args.step_ms}ms")
    for name, ms in results.items():
        if isinstance(ms, str):
            print(f"  {name:12} {ms}")
            continue
        per_step = ms * args.depth
        print(
            f"  {name:12} {ms:6.3f} ms/layer | {per_step:7.2f} ms/step (x{args.depth}) | "
            f"{per_step / args.step_ms * 100:5.1f}% of step"
        )
    if isinstance(results.get("flash-cudnn"), float):
        print(f"\n  fused headroom (naive - cudnn): {(naive_ms - results['flash-cudnn']) * args.depth:.2f} ms/step "
              f"({(naive_ms - results['flash-cudnn']) * args.depth / args.step_ms * 100:.1f}% of step)")


if __name__ == "__main__":
    main()
