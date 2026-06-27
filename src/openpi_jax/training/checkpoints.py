"""Checkpoint save/restore via Orbax."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save(checkpoint_dir: str | Path, step: int, state: Any) -> None:
    """Save a training state (params + opt_state + step) under ``checkpoint_dir/step``.

    TODO: wire up ``orbax.checkpoint.CheckpointManager`` with async saving and a retention policy. For now this
    raises so the train loop's no-op debug mode is explicit rather than silently dropping checkpoints.
    """
    import orbax.checkpoint as ocp

    ckptr = ocp.StandardCheckpointer()
    path = Path(checkpoint_dir).absolute() / str(step)
    ckptr.save(path, state)


def restore(checkpoint_dir: str | Path, step: int | None = None) -> Any:
    """Restore a training state. If ``step`` is None, restore the latest."""
    import orbax.checkpoint as ocp

    base = Path(checkpoint_dir).absolute()
    if step is None:
        steps = sorted(int(p.name) for p in base.iterdir() if p.name.isdigit())
        if not steps:
            raise FileNotFoundError(f"no checkpoints found under {base}")
        step = steps[-1]
    ckptr = ocp.StandardCheckpointer()
    return ckptr.restore(base / str(step))
