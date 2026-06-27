# Architecture

## π0 at a glance

π0 is a flow-matching vision-language-action model. It reuses a pretrained VLM (PaliGemma = SigLIP vision +
Gemma LLM) and bolts on a smaller **action expert** that denoises continuous action chunks.

### Token streams

A forward pass builds one sequence of tokens from three sources:

| Stream | Source | Processed by |
|---|---|---|
| Image tokens | SigLIP over each camera image | VLM (Gemma-2B) weights |
| Language tokens | SentencePiece prompt → embedding | VLM (Gemma-2B) weights |
| State + action tokens | proprioceptive state + noisy action chunk + flow time | Action expert (Gemma-300M) weights |

All tokens go through a **single shared self-attention** at every layer, but each token is processed by the
expert that owns it (a mixture-of-experts split by token *type*, not by routing). This lets the action tokens
attend to the full visual-language context while keeping the action pathway cheap.

### Attention mask

Block-causal:
- Prefix (images + text) attends bidirectionally within the prefix.
- Suffix (state + actions) attends to the entire prefix and causally within the suffix.

### Flow matching

Training uses the rectified-flow objective:

```
t ~ U(0,1),  e ~ N(0, I)
x_t = t·e + (1-t)·a            # interpolate between data action `a` and noise
u   = e - a                    # target velocity
loss = || v_θ(obs, x_t, t) - u ||²
```

Inference integrates `v_θ` from `x_1 ~ N(0, I)` to `x_0` with a handful of Euler steps (`num_inference_steps`),
yielding a chunk of `action_horizon` future actions executed open-loop before the next observation.

## π0-FAST

Same backbone, different action head: action chunks are discretized by the FAST tokenizer (DCT + entropy
coding) and decoded autoregressively like text. Simpler training (cross-entropy), slower inference.

## Code map

| Path | Responsibility |
|---|---|
| `models/siglip.py` | ViT image encoder |
| `models/gemma.py` | Gemma blocks, RMSNorm, RoPE, SwiGLU, expert configs |
| `models/action_expert.py` | state/action ↔ token projections, flow-time embedding |
| `models/pi0.py` | π0: embed prefix, shared trunk, flow-matching loss + sampling |
| `models/pi0_fast.py` | π0-FAST autoregressive variant |
| `training/` | config registry, optimizer, checkpoints, sharding, train loop |
| `data/`, `transforms/` | dataset iteration, normalization, embodiment transforms |
| `policies/` | `Policy` inference wrapper + websocket server |
