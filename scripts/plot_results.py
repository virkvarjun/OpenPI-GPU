"""Generate the Sharder result figures from the measured numbers (pandas + matplotlib).

All data below are REAL measurements on the runpod 4× H100 node (gemma_2b, bf16) — see FINDINGS.md /
STEP_BREAKDOWN.md for provenance. Run:  python scripts/plot_results.py   → writes PNGs to figures/.

Kept as data-in-code (not a CSV read) so the figures are reproducible from one file and the provenance of every
point is visible next to it.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)

# --- FSDP scaling (naive attention, fwd+bwd, sharding.fsdp_sharding) ---
strong = pd.DataFrame(
    {"gpus": [1, 2, 4], "ms": [195.6, 169.3, 135.5], "tflops": [344, 398, 497], "mfu": [34.7, 20.1, 12.5]}
)  # fixed global batch 4
weak = pd.DataFrame(
    {"gpus": [1, 2, 4], "ms": [195.5, 249.9, 274.6], "tflops": [344, 539, 980], "mfu": [34.8, 27.2, 24.8]}
)  # global batch 4*n_gpus (4 samples/GPU)

# --- Attention ablation (head_dim 256, ×18 layers, % of ~212ms step) ---
attn = pd.DataFrame(
    {"seq": [866, 2048, 4096], "naive_pct": [4.7, 22.2, 86.1], "flash_pct": [2.4, 8.3, 29.9],
     "naive_mem_gb": [0.14, 0.81, 3.22]}
)

# --- FlashAttention end-to-end (gemma_2b fwd+bwd, batch 4, 1×H100) ---
flash_e2e = pd.DataFrame({"impl": ["naive", "flash"], "ms": [195.85, 177.61], "mfu": [34.7, 38.3]})

# --- Comms/compute overlap: weak scaling, flash-default, default vs tuned XLA flags (scripts/fsdp_xla_flags.sh) ---
# 1 GPU has no collectives, so tuned == default there. Tuned gain grows with GPU count (more comms to hide).
overlap = pd.DataFrame(
    {
        "gpus": [1, 2, 4],
        "default_ms": [178.2, 232.3, 255.9],
        "tuned_ms": [178.2, 227.0, 246.0],
        "default_mfu": [38.1, 29.3, 26.6],
        "tuned_mfu": [38.1, 29.9, 27.6],
    }
)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    print(f"wrote figures/{name}")


def fig_scaling_time():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(strong.gpus, strong.ms, "o-", label="strong (fixed batch 4)")
    ax.plot(weak.gpus, weak.ms, "s-", label="weak (4 samples/GPU)")
    ax.axhline(195.5, ls="--", c="gray", label="ideal weak (flat)")
    ax.set(xlabel="GPUs", ylabel="device ms/step", title="FSDP scaling — step time (gemma_2b, H100)")
    ax.set_xticks([1, 2, 4])
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fsdp_scaling_time.png")


def fig_scaling_throughput():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(weak.gpus, weak.tflops, "s-", label="weak (measured)")
    ax.plot(strong.gpus, strong.tflops, "o-", label="strong (measured)")
    ax.plot([1, 2, 4], [344, 688, 1376], "k--", label="ideal linear")
    ax.set(xlabel="GPUs", ylabel="aggregate TFLOP/s", title="FSDP scaling — throughput")
    ax.set_xticks([1, 2, 4])
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fsdp_scaling_throughput.png")


def fig_mfu():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(strong.gpus, strong.mfu, "o-", label="strong")
    ax.plot(weak.gpus, weak.mfu, "s-", label="weak")
    ax.set(xlabel="GPUs", ylabel="per-GPU MFU (%)", title="FSDP scaling — per-GPU MFU")
    ax.set_xticks([1, 2, 4])
    ax.set_ylim(0, 40)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fsdp_scaling_mfu.png")


def fig_attention():
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(attn))
    w = 0.35
    ax.bar([i - w / 2 for i in x], attn.naive_pct, w, label="naive")
    ax.bar([i + w / 2 for i in x], attn.flash_pct, w, label="flash-xla")
    ax.axhspan(15, 20, color="orange", alpha=0.15, label="fuse-worthwhile gate")
    ax.set(xlabel="sequence length", ylabel="attention % of step", title="Attention cost vs sequence (gemma_2b)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(attn.seq)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "attention_vs_seq.png")


def fig_flash_e2e():
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(flash_e2e.impl, flash_e2e.ms, color=["#888", "#2a9d8f"])
    for b, mfu in zip(bars, flash_e2e.mfu, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f} ms\n{mfu:.1f}% MFU",
                ha="center", va="bottom")
    ax.set(ylabel="device ms/step", title="FlashAttention end-to-end (gemma_2b, 1×H100)")
    ax.set_ylim(0, 230)
    _save(fig, "flash_end_to_end.png")


def fig_overlap():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(overlap.gpus, overlap.default_ms, "o-", label="default XLA flags")
    ax.plot(overlap.gpus, overlap.tuned_ms, "s-", label="tuned overlap flags")
    ax.axhline(178.2, ls="--", c="gray", label="ideal weak (flat)")
    for g, dfl, tun in zip(overlap.gpus, overlap.default_ms, overlap.tuned_ms, strict=True):
        if dfl != tun:
            ax.annotate(f"-{(dfl - tun) / dfl * 100:.1f}%", (g, tun), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=9, color="#2a9d8f")
    ax.set(xlabel="GPUs", ylabel="device ms/step",
           title="Comms/compute overlap — weak scaling (flash attn)")
    ax.set_xticks([1, 2, 4])
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "comms_overlap.png")


if __name__ == "__main__":
    fig_scaling_time()
    fig_scaling_throughput()
    fig_mfu()
    fig_attention()
    fig_flash_e2e()
    fig_overlap()
    print(f"\nAll figures in {FIG}/")
