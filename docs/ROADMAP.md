# Roadmap

The scaffolding runs end-to-end on CPU with a fake data loader (see `examples/quickstart.py` and the smoke
tests). The items below turn it into a faithful OpenPI reimplementation. Rough order of dependency.

## 1. Modeling — the Gemma trunk (`models/gemma.py`, `models/pi0.py`)
- [ ] `GemmaBlock`: RMSNorm → grouped-query attention (RoPE, KV heads) → residual → RMSNorm → SwiGLU FFN.
- [ ] Dual-expert wiring in `Pi0._backbone`: VLM weights on prefix tokens, action-expert weights on suffix
      tokens, **shared** self-attention over the concatenation.
- [ ] Block-causal attention mask: prefix is bidirectional within itself; suffix attends to prefix and causally
      to suffix.
- [ ] Replace the identity pass-through in `Pi0._backbone`.

## 2. Vision (`models/siglip.py`)
- [ ] Confirm SigLIP-So400m hyperparameters; add attention-pooling head if porting full PaliGemma.

## 3. Weight porting (`docs/CHECKPOINTS.md`)
- [ ] Loader to map PaliGemma / OpenPI checkpoint tensors → this repo's Flax param tree.

## 4. Data (`data/dataset.py`, `transforms/`)
- [ ] `LeRobotDataLoader`: real LeRobot dataset iteration, action chunking, image decode/resize.
- [ ] Per-embodiment input/output transforms (ALOHA, DROID, Libero).
- [ ] Normalization-stats computation + caching to `assets/<repo_id>/norm_stats.json`.

## 5. π0-FAST (`models/pi0_fast.py`, `models/tokenizer.py`)
- [ ] FAST tokenizer: DCT + BPE/entropy coding `encode`/`decode`.
- [ ] Autoregressive training (cross-entropy) and KV-cache decode.

## 6. Training at scale (`training/sharding.py`, `training/checkpoints.py`)
- [ ] FSDP sharding across a device mesh; shard params + data.
- [ ] Orbax `CheckpointManager` with async save + retention.
- [ ] Mixed precision, gradient accumulation, EMA (optional).

## 7. Serving (`policies/serve.py`)
- [ ] `build_policy`: restore params, assemble transform stacks.
- [ ] Action-chunk execution / temporal ensembling on the client side.

## 8. Eval
- [ ] Sim eval harness (ALOHA sim / Libero) reporting success rates.
