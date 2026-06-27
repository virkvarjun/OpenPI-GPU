"""M4 correctness: multi-process data parity using the REAL assembly primitive.

`data_sharding_test.py` proves the sharding *indices* are correct. This file proves the end-to-end data path:
each process loads its within-batch shard, `jax.make_array_from_process_local_data` reassembles the global batch,
and the result is **numerically identical** to what a single-process run would feed the model — including the
gradient. That is the "multi-process matches single-process learning" guarantee, demonstrated without the heavy
pi0 model (a tiny jitted loss stands in; the data path is what's under test).

Single-process assembly runs in-process; the true multi-process case is exercised via localhost processes
(`scripts/launch_local.py`).
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import data_sharding, sharding

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"

# Toy problem shared by reference and sharded paths.
_N, _B, _D, _SEED, _STEPS = 256, 16, 8, 11, 4


def _row(i: int) -> np.ndarray:
    """Deterministic dataset row for index i (same on every process)."""
    return np.random.default_rng(1000 + int(i)).standard_normal(_D).astype(np.float32)


def _W() -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(99).standard_normal((_D, _D)).astype(np.float32))


def _loss_and_grad(batch: jnp.ndarray, w: jnp.ndarray):
    def loss(w_):
        return jnp.mean((batch @ w_) ** 2)

    return jax.value_and_grad(loss)(w)


def _reference_batch(step: int) -> np.ndarray:
    idx = data_sharding.global_batch_indices(_N, _B, step, shuffle=True, seed=_SEED)
    return np.stack([_row(i) for i in idx])


def test_single_process_assembly_matches_reference():
    """With one process/device, the assembled batch + its gradient equal the reference exactly."""
    mesh = sharding.make_mesh(num_fsdp_devices=1)
    data_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    w = _W()

    pc, pi = jax.process_count(), jax.process_index()
    local = _B // pc
    stream = data_sharding.process_index_stream(_N, _B, pi, pc, shuffle=True, seed=_SEED)

    for s in range(_STEPS):
        local_idx = stream[s * local : (s + 1) * local]
        local_batch = np.stack([_row(i) for i in local_idx])
        assembled = jax.make_array_from_process_local_data(data_sh, local_batch)

        ref = _reference_batch(s)
        np.testing.assert_array_equal(np.asarray(assembled), ref)
        la, ga = _loss_and_grad(assembled, w)
        lr, gr = _loss_and_grad(jnp.asarray(ref), w)
        np.testing.assert_allclose(la, lr, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(ga), np.asarray(gr), rtol=1e-6, atol=1e-6)


def test_multiprocess_assembly_matches_reference(tmp_path):
    """2 processes x 1 device: each process's assembled batch + grad equal the single-process reference."""
    nproc = 2
    probe = tmp_path / "parity_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import json, pathlib
            import jax, jax.numpy as jnp, numpy as np
            from openpi.training import data_sharding, distributed, sharding

            # Must run before ANY jax computation (mirrors train.py:main where it is the first line).
            distributed.maybe_initialize()

            N, B, D, SEED, STEPS = 256, 16, 8, 11, 4
            row = lambda i: np.random.default_rng(1000 + int(i)).standard_normal(D).astype(np.float32)
            W = jnp.asarray(np.random.default_rng(99).standard_normal((D, D)).astype(np.float32))
            def lg(batch):
                return jax.value_and_grad(lambda w: jnp.mean((batch @ w) ** 2))(W)

            mesh = sharding.make_mesh(num_fsdp_devices=1)
            data_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
            pc, pi = jax.process_count(), jax.process_index()
            local = B // pc
            stream = data_sharding.process_index_stream(N, B, pi, pc, shuffle=True, seed=SEED)

            ok = True
            for s in range(STEPS):
                li = stream[s * local : (s + 1) * local]
                lb = np.stack([row(i) for i in li])
                assembled = jax.make_array_from_process_local_data(data_sh, lb)
                ref_idx = data_sharding.global_batch_indices(N, B, s, shuffle=True, seed=SEED)
                ref = np.stack([row(i) for i in ref_idx])
                # This process's shard of the global batch == its within-batch slice of the reference.
                ok &= np.array_equal(np.asarray(assembled.addressable_shards[0].data), lb)
                ok &= np.array_equal(lb, ref[pi * local : (pi + 1) * local])
                # Loss/grad are global reductions over the assembled batch -> replicated -> fetchable. They must
                # equal the single-process reference (the "didn't break learning" guarantee).
                la, ga = lg(assembled); lr, gr = lg(jnp.asarray(ref))
                ok &= bool(np.allclose(float(la), float(lr), rtol=1e-5, atol=1e-5))
                ok &= bool(np.allclose(np.asarray(ga), np.asarray(gr), rtol=1e-5, atol=1e-5))

            pathlib.Path(OUT_DIR, f"parity_{pi}.json").write_text(json.dumps({"pid": int(pi), "ok": bool(ok)}))
            """
        ).replace("OUT_DIR", repr(str(tmp_path)))
    )

    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "launch_local.py"),
        "--nproc",
        str(nproc),
        "--devices-per-proc",
        "1",
        "--",
        sys.executable,
        str(probe),
    ]
    res = subprocess.run(
        cmd, env={**os.environ, "PYTHONPATH": str(_SRC)}, capture_output=True, text=True, timeout=240
    )
    assert res.returncode == 0, f"launch failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    for i in range(nproc):
        r = json.loads((tmp_path / f"parity_{i}.json").read_text())
        assert r["ok"], f"process {i} parity failed"
