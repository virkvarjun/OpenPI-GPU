"""Smoke test for the M5 scaling-study probe (one device-count measurement)."""

import json
import os
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]


def test_probe_runs_and_reports_at_two_devices():
    res = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "scaling_study.py"), "--probe"],
        env={
            **os.environ,
            "PYTHONPATH": str(_REPO / "src"),
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["devices"] == 2
    assert out["global_batch"] == 16  # weak scaling: 8 * 2
    assert out["median_step_ms"] > 0
    assert out["achieved_tflops"] > 0
