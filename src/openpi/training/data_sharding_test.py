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
