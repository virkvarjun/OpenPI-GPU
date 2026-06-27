# STEP_BREAKDOWN — openpi pi0 training step (attribution)

Real `openpi.models.pi0` step (gemma naive-softmax attention + FFN + action expert + flow-matching
loss) through a faithful nnx value_and_grad + AdamW step, attributed by HLO named-scope + op type.

> ⚠️ **Cheap-ladder run: `dummy` gemma variant (width=64, depth=4) on CPU.** The code path is the real
> model, but the proportions are NOT production: tiny dims + CPU inflate data-wait/other (per-op dispatch)
> and shrink the GEMM/attention share. Production proportions are HARDWARE-GATED — collect them by running
> this attribution on train.py's ptrain_step on a real accelerator slice.

Window: 8 steps | wall 15912.668 ms/step | device-busy 4547.263 ms/step

| category | device ms/step | % wall |
|----------|---------------:|-------:|
| attention | 143.4074 | 0.9% |
| matmul-FFN | 218.0563 | 1.4% |
| collectives | 0.0000 | 0.0% |
| optimizer | 679.8666 | 4.3% |
| other | 3505.9326 | 22.0% |
| data-wait | 11365.4048 | 71.4% |

**Dominant (this run): `data-wait`** (71.4% of wall).

## Decision gate (apply to PRODUCTION proportions, not these CPU/dummy ones)
- **data-wait dominates** → input pipeline: image decode, prefetch depth, H2D overlap.
- **collectives dominate** (multi-host) → comms/compute overlap, collective placement, mesh/fsdp tuning.
- **attention > ~15-20%** → fused attention (jax.nn.dot_product_attention flash; then Pallas/splash). Do NOT touch saturated matmul/GEMM.
- **memory-bound / OOM** → tune the remat policy off `nothing_saveable` to recover recompute.

On real hardware the GEMM/attention share rises sharply (large dims) and data-wait/other shrink; the
instrument is unchanged. This artifact + the harness are the V2/M1 deliverable; the optimization target
is chosen from the production breakdown.
