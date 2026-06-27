# OpenPI-JAX

A from-scratch **JAX / Flax** recreation of [OpenPI](https://github.com/Physical-Intelligence/openpi) — the
open-source **π0** and **π0-FAST** vision-language-action (VLA) models for robot learning.

This repo reimplements the model architectures, the flow-matching / autoregressive action heads, the training
loop, normalization, and a policy-serving server, with a focus on a clean, readable, hackable codebase.

> Status: **scaffolding**. Module layout and interfaces are in place; implementations are stubbed with `TODO`s
> and minimal smoke tests. See [docs/ROADMAP.md](docs/ROADMAP.md).

## What's here

| Component | Description |
|---|---|
| **π0** | Flow-matching VLA: PaliGemma (SigLIP + Gemma) backbone + a Gemma "action expert" that denoises action chunks. |
| **π0-FAST** | Autoregressive VLA using FAST action tokenization over the same backbone. |
| Vision | SigLIP image encoder. |
| Backbone | Gemma decoder-only LLM with a mixture-of-experts-style dual-stream attention (VLM + action expert). |
| Training | Config-driven trainer (`tyro` CLIs), Optax optimizer, Orbax checkpoints. |
| Data | LeRobot-dataset loader, data transforms, per-dataset normalization stats. |
| Serving | Websocket policy server for closed-loop robot inference. |

## Architecture (π0)

```
  images ─► SigLIP ─┐
                    ├─► Gemma (VLM expert) ─┐
  language ─► embed ┘                       │ shared attention
                                            ├─► flow-matching velocity ─► action chunk
  state + noisy actions ─► Gemma (action expert) ─┘
```

π0 predicts a chunk of future actions by integrating a learned velocity field (flow matching / rectified flow)
conditioned on observations. π0-FAST instead autoregressively decodes FAST-tokenized actions.

## Quickstart

```bash
# Create the environment (Python 3.11+). uv recommended.
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"           # CPU
# uv pip install -e ".[cuda,dev]"    # NVIDIA GPU

# Run the smoke tests
pytest

# Inspect / list training configs
openpi-train --help

# Train (example config — see src/openpi_jax/training/config.py)
openpi-train --config pi0_aloha_sim

# Serve a trained policy over websocket
openpi-serve --config pi0_aloha_sim --checkpoint checkpoints/pi0_aloha_sim/latest
```

## Repository layout

```
src/openpi_jax/
  models/        pi0, pi0_fast, gemma backbone, siglip vision, action expert, tokenizer
  training/      config registry, train loop, optimizer, checkpoint utils, sharding
  data/          dataset loading, batching, transforms
  transforms/    input/output data transforms (robot-specific)
  policies/      Policy wrapper + websocket serving
  shared/        normalization, array typing, pytree utils
scripts/         thin CLI entrypoints
tests/           smoke + unit tests
examples/        end-to-end usage notebooks/scripts
docs/            design notes & roadmap
```

## Relationship to OpenPI

This is an independent reimplementation for learning and research. Model definitions follow the publicly
described π0 architecture (Black et al., 2024). Pretrained weights are **not** distributed here; see
[docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) for porting notes.

## License

Apache-2.0. See [LICENSE](LICENSE).
