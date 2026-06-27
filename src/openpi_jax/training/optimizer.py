"""Optimizer construction (Optax): AdamW with warmup + cosine decay and global-norm clipping."""

from __future__ import annotations

import optax

from openpi_jax.training.config import OptimizerConfig


def make_lr_schedule(cfg: OptimizerConfig) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.peak_lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.decay_steps,
        end_value=cfg.peak_lr * 0.1,
    )


def make_optimizer(cfg: OptimizerConfig) -> optax.GradientTransformation:
    schedule = make_lr_schedule(cfg)
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.b1,
            b2=cfg.b2,
            weight_decay=cfg.weight_decay,
        ),
    )
