"""Unit tests for the attribution classifier + Breakdown math (no model/trace needed)."""

from openpi.training import attribution


def test_classify_collectives():
    assert attribution.classify("all-reduce.3", "jit(step)/.../fsdp") == "collectives"
    assert attribution.classify("reduce-scatter", "") == "collectives"
    assert attribution.classify("all-gather.1", "") == "collectives"


def test_classify_attention_softmax():
    # softmax ops under an attention scope -> attention (the headroom per the audit).
    assert attribution.classify("subtract_exponential_fusion", "jit/.../attention/exp") == "attention"
    assert attribution.classify("reduce.20", "jit/.../attention/reduce_max") == "attention"


def test_classify_matmul_and_ffn():
    assert attribution.classify("dot.38", "jit/.../attention/dot_general") == "matmul-FFN"  # attn GEMM stays matmul
    assert attribution.classify("dot.5", "jit/.../mlp/dot_general") == "matmul-FFN"
    assert attribution.classify("multiply_fusion", "jit/.../mlp/mul") == "matmul-FFN"


def test_classify_optimizer_and_other():
    assert attribution.classify("add.1", "jit/.../adam/scale_by_adam") == "optimizer"
    assert attribution.classify("copy.2", "jit/.../something_else") == "other"


def test_breakdown_percentages_and_dominant():
    bd = attribution.Breakdown(
        category_us={"attention": 100.0, "matmul-FFN": 300.0, "collectives": 0.0, "optimizer": 50.0, "other": 50.0},
        device_busy_us=500.0,
        wall_us=1000.0,
        n_steps=10,
    )
    pct = bd.percentages()
    assert abs(sum(pct.values()) - 1.0) < 1e-9  # categories + data-wait sum to 1
    assert abs(pct["data-wait"] - 0.5) < 1e-9  # 1000 wall - 500 busy
    assert abs(pct["matmul-FFN"] - 0.3) < 1e-9
    assert bd.dominant() == "data-wait"
    assert "matmul-FFN" in bd.table()
