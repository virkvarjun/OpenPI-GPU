"""Tests for G1 cluster bring-up.

Two layers:
  - Unit: `maybe_initialize` is a no-op (returns False) for a single-process launch, without ever calling into
    the JAX distributed runtime.
  - Integration (cheap ladder): spawn N localhost processes via `scripts/launch_local.py`, have each report its
    view of the cluster, and assert `jax.device_count()` is GLOBAL (== N·devices_per_proc), process indices are
    distinct, and the mesh from `sharding.make_mesh` spans all processes' devices.
"""

import json
import pathlib
import subprocess
import sys
import textwrap

from openpi.training import distributed

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"


def test_single_process_is_noop():
    # No rendezvous env => no-op, and we never touch jax.distributed.
    assert distributed.maybe_initialize({}) is False
    # Partial env (missing process_id) is also a no-op, not an error.
    assert distributed.maybe_initialize({"JAX_COORDINATOR_ADDRESS": "localhost:1234", "JAX_NUM_PROCESSES": "2"}) is False


def test_multiprocess_device_count_is_global(tmp_path):
    """Launch 2 processes x 1 device and assert the cluster view is global."""
    nproc = 2
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            import json, os, pathlib
            import jax
            from openpi.training import distributed, sharding
            active = distributed.maybe_initialize()
            mesh = sharding.make_mesh(num_fsdp_devices=1)
            out = {{
                "active": active,
                "pid": jax.process_index(),
                "nproc": jax.process_count(),
                "global_dev": jax.device_count(),
                "local_dev": jax.local_device_count(),
                "mesh_size": int(mesh.devices.size),
            }}
            pathlib.Path(r"{tmp_path}", f"out_{{jax.process_index()}}.json").write_text(json.dumps(out))
            """
        )
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
    env = {"PYTHONPATH": str(_SRC)}
    import os

    full_env = {**os.environ, **env}
    res = subprocess.run(cmd, env=full_env, capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, f"launch failed:\nstdout={res.stdout}\nstderr={res.stderr}"

    results = [json.loads((tmp_path / f"out_{i}.json").read_text()) for i in range(nproc)]
    pids = sorted(r["pid"] for r in results)
    assert pids == list(range(nproc))  # distinct process indices
    for r in results:
        assert r["active"] is True
        assert r["nproc"] == nproc
        assert r["global_dev"] == nproc  # device_count() is GLOBAL across processes
        assert r["local_dev"] == 1
        assert r["mesh_size"] == nproc  # make_mesh spans all processes' devices
