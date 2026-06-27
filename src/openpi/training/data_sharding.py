"""G3: deterministic, per-process *within-batch* dataset sharding for the map-style (LeRobot) loader.

The design goal is **multi-process parity with single-process**: the set of examples in every global optimization
step must be identical no matter how many processes we split it across, so learning is unchanged.

Key idea — shard *within each global batch*, not across the whole dataset. Given a deterministic global index
order (a seeded permutation when shuffling, else sequential), global batch ``s`` is ``order[s·B : (s+1)·B]``.
Process ``k`` of ``N`` takes the contiguous sub-slice ``order[s·B + k·local : s·B + (k+1)·local]`` where
``local = B // N``. When each process loads its rows and the global array is reassembled in process-index order
(via ``jax.make_array_from_process_local_data``), the result is *exactly* ``order[s·B : (s+1)·B]`` — the same
batch a single-process run would see. This is why parity holds to bit precision.

Contrast with naive ``order[k::N]`` striping (e.g. openpi-comet): that interleaves across the whole dataset, so
the per-step global batch composition differs from single-process and parity breaks. See PLAN.md (G3).

This module is intentionally pure NumPy (no jax/torch), so the sharding math is unit-testable on its own.
"""

from __future__ import annotations

import numpy as np


def global_order(dataset_len: int, *, shuffle: bool, seed: int) -> np.ndarray:
    """The deterministic global example order shared by all processes for one epoch.

    Same ``seed`` ⇒ same order on every process (required so the within-batch shards line up).
    """
    if shuffle:
        return np.random.default_rng(seed).permutation(dataset_len)
    return np.arange(dataset_len)


def num_global_batches(dataset_len: int, global_batch_size: int) -> int:
    """Number of full global batches in one epoch (drop_last semantics, matching the torch loader)."""
    return dataset_len // global_batch_size


def process_index_stream(
    dataset_len: int,
    global_batch_size: int,
    process_index: int,
    process_count: int,
    *,
    shuffle: bool,
    seed: int,
) -> np.ndarray:
    """Flat sequence of dataset indices this process should consume, in order, over one epoch.

    Batching the returned indices into groups of ``global_batch_size // process_count`` yields this process's
    per-step local batches; concatenating those across processes in index order reconstructs each global batch.
    """
    if global_batch_size % process_count != 0:
        raise ValueError(f"global_batch_size ({global_batch_size}) must be divisible by process_count ({process_count})")
    if not 0 <= process_index < process_count:
        raise ValueError(f"process_index {process_index} out of range for process_count {process_count}")

    order = global_order(dataset_len, shuffle=shuffle, seed=seed)
    n_batches = num_global_batches(dataset_len, global_batch_size)
    local = global_batch_size // process_count

    segments = []
    for s in range(n_batches):
        base = s * global_batch_size + process_index * local
        segments.append(order[base : base + local])
    return np.concatenate(segments) if segments else np.empty(0, dtype=order.dtype)


def global_batch_indices(
    dataset_len: int, global_batch_size: int, step: int, *, shuffle: bool, seed: int
) -> np.ndarray:
    """The indices a single-process run would put in global batch ``step`` — the parity reference."""
    order = global_order(dataset_len, shuffle=shuffle, seed=seed)
    return order[step * global_batch_size : (step + 1) * global_batch_size]
