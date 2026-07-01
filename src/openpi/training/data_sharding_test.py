"""Tests for G3 within-batch sharding math (pure, no jax/torch needed).

Covers the V1 data guarantees at the index level: determinism, no-duplication + full coverage across processes,
and — the important one — *within-batch reconstruction* (the foundation of single-vs-multi parity).
"""

import numpy as np
import pytest

from openpi.training import data_sharding


def test_determinism_same_seed_same_stream():
    a = data_sharding.process_index_stream(1000, 32, 1, 4, shuffle=True, seed=7)
    b = data_sharding.process_index_stream(1000, 32, 1, 4, shuffle=True, seed=7)
    assert np.array_equal(a, b)
    c = data_sharding.process_index_stream(1000, 32, 1, 4, shuffle=True, seed=8)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("process_count", [1, 2, 4, 8])
def test_no_duplication_and_full_coverage(process_count):
    n, b = 1000, 32
    streams = [
        data_sharding.process_index_stream(n, b, k, process_count, shuffle=True, seed=0)
        for k in range(process_count)
    ]
    consumed = np.concatenate(streams)
    # No example appears twice across processes within the epoch's full batches.
    assert len(consumed) == len(set(consumed.tolist()))
    # Coverage == exactly the indices in the full global batches (drop_last tail excluded).
    n_batches = data_sharding.num_global_batches(n, b)
    expected = data_sharding.global_order(n, shuffle=True, seed=0)[: n_batches * b]
    assert set(consumed.tolist()) == set(expected.tolist())


@pytest.mark.parametrize("process_count", [1, 2, 4])
def test_within_batch_reconstruction_matches_single_process(process_count):
    """Concatenating per-process local batches (process order) reproduces the single-process global batch."""
    n, b = 256, 32
    seed, shuffle = 3, True
    local = b // process_count
    streams = [
        data_sharding.process_index_stream(n, b, k, process_count, shuffle=shuffle, seed=seed)
        for k in range(process_count)
    ]
    for s in range(data_sharding.num_global_batches(n, b)):
        # Each process's local batch for step s is its stream slice [s*local:(s+1)*local].
        per_proc = [streams[k][s * local : (s + 1) * local] for k in range(process_count)]
        reassembled = np.concatenate(per_proc)
        reference = data_sharding.global_batch_indices(n, b, s, shuffle=shuffle, seed=seed)
        assert np.array_equal(reassembled, reference)


def test_single_process_is_plain_order():
    # process_count==1 => the stream is exactly the global order over full batches (no surprises).
    n, b = 100, 10
    stream = data_sharding.process_index_stream(n, b, 0, 1, shuffle=False, seed=0)
    assert np.array_equal(stream, np.arange(100))


def test_rejects_indivisible_batch():
    with pytest.raises(ValueError):
        data_sharding.process_index_stream(100, 30, 0, 4, shuffle=False, seed=0)


# --- G4: epoch-aware seeding + exact resume ---


def test_epoch_seed_zero_is_identity_and_reshuffles():
    assert data_sharding.epoch_seed(42, 0) == 42  # epoch 0 == seed (backward compatible)
    o0 = data_sharding.global_order(100, shuffle=True, seed=42, epoch=0)
    o1 = data_sharding.global_order(100, shuffle=True, seed=42, epoch=1)
    assert not np.array_equal(o0, o1)  # different epochs give a different shuffle
    assert np.array_equal(o1, data_sharding.global_order(100, shuffle=True, seed=42, epoch=1))  # reproducible


def test_resume_position_maps_step_to_epoch_offset():
    # 100 examples / batch 10 -> 10 batches/epoch. 23 consumed -> epoch 2, offset 3.
    assert data_sharding.resume_position(23, 100, 10) == (2, 3)
    assert data_sharding.resume_position(0, 100, 10) == (0, 0)
    assert data_sharding.resume_position(10, 100, 10) == (1, 0)  # exact epoch boundary


def test_start_batch_skips_leading_batches():
    n, b, pc, pi = 100, 10, 2, 0
    local = b // pc
    full = data_sharding.process_index_stream(n, b, pi, pc, shuffle=True, seed=0, epoch=2)
    resumed = data_sharding.process_index_stream(n, b, pi, pc, shuffle=True, seed=0, epoch=2, start_batch=3)
    assert np.array_equal(resumed, full[3 * local :])  # skipped exactly 3 batches


def test_resume_continuity_no_repeat_or_skip_across_epoch_boundary():
    """Resuming from the checkpointed step count continues the stream exactly (no repeated/skipped examples)."""
    n, b, pc, pi = 60, 12, 3, 1
    local = b // pc
    uninterrupted = np.concatenate(
        [
            data_sharding.process_index_stream(n, b, pi, pc, shuffle=True, seed=7, epoch=0),
            data_sharding.process_index_stream(n, b, pi, pc, shuffle=True, seed=7, epoch=1),
        ]
    )
    consumed_batches = 7  # crashed after 7 global steps
    e, off = data_sharding.resume_position(consumed_batches, n, b)
    assert (e, off) == (1, 2)
    resumed = data_sharding.process_index_stream(n, b, pi, pc, shuffle=True, seed=7, epoch=e, start_batch=off)
    # the resumed stream == exactly the tail of the uninterrupted run after `consumed_batches` batches
    assert np.array_equal(resumed, uninterrupted[consumed_batches * local :])
