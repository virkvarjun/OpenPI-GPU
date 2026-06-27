# Checkpoints & weights

Pretrained weights are **not** committed to this repo (they're large and separately licensed). This page tracks
how to obtain and port them.

## Layout

Place downloaded assets under `assets/` (git-ignored):

```
assets/
  paligemma/
    tokenizer.model          # SentencePiece model for PaligemmaTokenizer
    params/                  # ported VLM weights (Flax msgpack / Orbax)
  <repo_id>/
    norm_stats.json          # per-dataset normalization stats (Normalizer.save)
```

## Porting PaliGemma → OpenPI-JAX

The VLM backbone (SigLIP + Gemma-2B) can be initialized from the public PaliGemma checkpoint. A porting script
(TODO, see ROADMAP §3) maps source tensors into this repo's Flax param tree:

- SigLIP: patch-embed conv, position embedding, per-block LN/attention/MLP, post-LN.
- Gemma: per-layer RMSNorm scales, QKV/O projections, gating/up/down FFN, final norm, embedding table.

The action expert (Gemma-300M) is trained from scratch (random init) unless porting a full π0 checkpoint.

## Normalization stats

Compute once per dataset and cache:

```python
from openpi_jax.shared.normalize import NormStats, Normalizer
# ... accumulate mean/std (or q01/q99) over the dataset ...
Normalizer(stats, mode="quantile").save("assets/<repo_id>/norm_stats.json")
```

Stats are loaded at train and serve time so the model always sees unit-scale inputs.
