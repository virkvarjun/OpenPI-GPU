# STEP_BREAKDOWN — openpi pi0 training step (attribution)

Real `openpi.models.pi0` step (gemma naive-softmax attention + FFN + action expert + flow-matching
loss) through a faithful nnx value_and_grad + AdamW step, attributed by HLO named-scope + op type.

> ⚠️ **Cheap-ladder run: `dummy` gemma variant (width=64, depth=4) on CPU.** The code path is the real
> model, but the proportions are NOT production: tiny dims + CPU inflate data-wait/other (per-op dispatch)
> and shrink the GEMM/attention share. Production proportions are HARDWARE-GATED — collect them by running
> this attribution on train.py's ptrain_step on a real accelerator slice.

Window: 8 steps | wall 13154.037 ms/step | device-busy 2856.452 ms/step

| category | device ms/step | % wall |
|----------|---------------:|-------:|
| vision | 163.3814 | 1.2% |
| image-aug | 3.8081 | 0.0% |
| embedding | 1.1324 | 0.0% |
| attention | 75.5700 | 0.6% |
| matmul-FFN | 8.4277 | 0.1% |
| collectives | 0.0000 | 0.0% |
| optimizer | 0.0000 | 0.0% |
| other | 2604.1319 | 19.8% |
| data-wait | 10297.5855 | 78.3% |

**Dominant (this run): `data-wait`** (78.3% of wall).

## Decision gate (apply to PRODUCTION proportions, not these CPU/dummy ones)
- **data-wait dominates** → input pipeline: image decode, prefetch depth, H2D overlap.
- **collectives dominate** (multi-host) → comms/compute overlap, collective placement, mesh/fsdp tuning.
- **attention > ~15-20%** → fused attention (jax.nn.dot_product_attention flash; then Pallas/splash). Do NOT touch saturated matmul/GEMM.
- **memory-bound / OOM** → tune the remat policy off `nothing_saveable` to recover recompute.

On real hardware the GEMM/attention share rises sharply (large dims) and data-wait/other shrink; the
instrument is unchanged. This artifact + the harness are the V2/M1 deliverable; the optimization target
is chosen from the production breakdown.
