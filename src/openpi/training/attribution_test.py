"""Unit + integration tests for the attribution classifier and Breakdown math.

Classifier cases use the REAL pi0 scope vocabulary (encoderblock/attn/dot_general/pixelwise/...). The collectives
case is an integration test: a data-parallel step across 2 localhost processes whose gradient all-reduce must
show up in the breakdown's `collectives` bucket.
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

from openpi.training import attribution

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"


def test_classify_collectives():
    assert attribution.classify("all-reduce.3", "jit(step)/.../fsdp") == "collectives"
    assert attribution.classify("reduce-scatter", "") == "collectives"
    assert attribution.classify("all-gather.1", "") == "collectives"


def test_classify_vision_and_image_aug():
    # SigLIP encoder (incl. its own MHA softmax) -> vision, not gemma attention.
    assert attribution.classify("exponential.2", "jit/.../img/encoderblock/MultiHeadDotProductAttention_0/softmax") == "vision"
    assert attribution.classify("dot.9", "jit/.../encoderblock/MlpBlock_0/dot_general") == "vision"
    # augmax on-device augmentation -> image-aug (was the bulk of old "other").
    assert attribution.classify("fusion.4", "jit/.../vmap(jit(pixelwise))/mul") == "image-aug"
    assert attribution.classify("gather.1", "jit/.../map_coordinates/...") == "image-aug"


def test_classify_embedding():
    assert attribution.classify("dynamic-slice.1", "jit/.../embedder/take") == "embedding"
    assert attribution.classify("add.2", "jit/.../pos_embed/add") == "embedding"


def test_classify_gemma_attention_softmax_vs_matmul():
    # gemma attention softmax -> attention; the QK/AV dots -> matmul-FFN (XLA saturates them).
    assert attribution.classify("subtract_exponential_fusion", "jit/.../Transformer/layers/attn/exp") == "attention"
    assert attribution.classify("dot.38", "jit/.../Transformer/layers/attn/dot_general") == "matmul-FFN"


def test_classify_matmul_optimizer_other():
    assert attribution.classify("dot.5", "jit/.../Transformer/layers/mlp/dot_general") == "matmul-FFN"
    assert attribution.classify("add.1", "jit/.../adam/scale_by_adam") == "optimizer"
    assert attribution.classify("copy.2", "jit/.../something_unscoped") == "other"


def test_breakdown_percentages_and_dominant():
    bd = attribution.Breakdown(
        category_us={"vision": 200.0, "matmul-FFN": 200.0, "attention": 50.0, "optimizer": 50.0},
        device_busy_us=500.0,
        wall_us=1000.0,
        n_steps=10,
    )
    pct = bd.percentages()
    assert abs(sum(pct.values()) - 1.0) < 1e-9  # all categories + data-wait sum to 1
    assert abs(pct["data-wait"] - 0.5) < 1e-9
    assert abs(pct["vision"] - 0.2) < 1e-9
    assert bd.dominant() == "data-wait"
    assert "vision" in bd.table()


def test_multiprocess_grad_all_reduce_shows_collectives(tmp_path):
    """2 processes x 1 device: a data-parallel step's gradient all-reduce appears in the collectives bucket."""
    nproc = 2
    probe = tmp_path / "coll_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import json, pathlib, tempfile
            import jax, jax.numpy as jnp, numpy as np
            from openpi.training import attribution, distributed, sharding

            distributed.maybe_initialize()  # before any jax compute
            mesh = sharding.make_mesh(num_fsdp_devices=1)  # (data=2, fsdp=1) across the 2 processes
            data_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS, None))
            repl = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

            rng = np.random.default_rng(0)
            x = jax.device_put(rng.standard_normal((64, 128), dtype=np.float32), data_sh)  # batch sharded over hosts
            w = jax.device_put(rng.standard_normal((128, 128), dtype=np.float32), repl)     # replicated params

            # grad of a global mean over the host-sharded batch -> requires an all-reduce across the data axis.
            # x must be PASSED (not closed over) since it spans processes.
            step = jax.jit(jax.value_and_grad(lambda w, x: jnp.mean((x @ w) ** 2), argnums=0),
                           in_shardings=(repl, data_sh), out_shardings=(repl, repl))
            compiled = step.lower(w, x).compile()
            bd = attribution.attribute_step(lambda: step(w, x), compiled, trace_dir=tempfile.mkdtemp(), warmup=2, iters=10)
            pathlib.Path(OUT, f"coll_{jax.process_index()}.json").write_text(
                json.dumps({"pid": int(jax.process_index()), "collectives_us": bd.category_us.get("collectives", 0.0)})
            )
            """
        ).replace("OUT", repr(str(tmp_path)))
    )
    cmd = [
        sys.executable, str(_REPO / "scripts" / "launch_local.py"),
        "--nproc", str(nproc), "--devices-per-proc", "1", "--", sys.executable, str(probe),
    ]
    res = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(_SRC)}, capture_output=True, text=True, timeout=240)
    assert res.returncode == 0, f"launch failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    results = [json.loads((tmp_path / f"coll_{i}.json").read_text()) for i in range(nproc)]
    # At least one process must observe collective time from the gradient all-reduce.
    assert any(r["collectives_us"] > 0 for r in results), f"no collectives observed: {results}"
