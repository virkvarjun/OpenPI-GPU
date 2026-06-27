"""Per-dataset normalization statistics and (de)normalization.

Robot action/state distributions vary wildly across embodiments, so OpenPI normalizes inputs to a roughly
unit-scale space before the model sees them, and de-normalizes the model's outputs back to physical units.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class NormStats:
    """Running statistics for one named array (e.g. ``"actions"`` or ``"state"``)."""

    mean: np.ndarray
    std: np.ndarray
    # Optional quantiles for quantile-based normalization (π0 uses q01/q99 for some embodiments).
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None

    def to_dict(self) -> dict:
        d = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        if self.q01 is not None:
            d["q01"] = self.q01.tolist()
        if self.q99 is not None:
            d["q99"] = self.q99.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> NormStats:
        def arr(key):
            return np.asarray(d[key], dtype=np.float32) if key in d and d[key] is not None else None

        return cls(mean=arr("mean"), std=arr("std"), q01=arr("q01"), q99=arr("q99"))


class Normalizer:
    """Applies and inverts normalization for a dict of named arrays.

    Two modes are supported, matching OpenPI:
      - ``"mean_std"``: ``(x - mean) / (std + eps)``
      - ``"quantile"``: maps ``[q01, q99]`` to ``[-1, 1]``
    """

    def __init__(self, stats: dict[str, NormStats], mode: str = "mean_std", eps: float = 1e-6):
        if mode not in ("mean_std", "quantile"):
            raise ValueError(f"unknown normalization mode: {mode}")
        self.stats = stats
        self.mode = mode
        self.eps = eps

    def normalize(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(data)
        for key, s in self.stats.items():
            if key not in out:
                continue
            if self.mode == "mean_std":
                out[key] = (out[key] - s.mean) / (s.std + self.eps)
            else:  # quantile
                out[key] = 2.0 * (out[key] - s.q01) / (s.q99 - s.q01 + self.eps) - 1.0
        return out

    def unnormalize(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(data)
        for key, s in self.stats.items():
            if key not in out:
                continue
            if self.mode == "mean_std":
                out[key] = out[key] * (s.std + self.eps) + s.mean
            else:  # quantile
                out[key] = (out[key] + 1.0) / 2.0 * (s.q99 - s.q01 + self.eps) + s.q01
        return out

    # --- serialization ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mode": self.mode, "eps": self.eps, "stats": {k: v.to_dict() for k, v in self.stats.items()}}
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        payload = json.loads(Path(path).read_text())
        stats = {k: NormStats.from_dict(v) for k, v in payload["stats"].items()}
        return cls(stats=stats, mode=payload.get("mode", "mean_std"), eps=payload.get("eps", 1e-6))
