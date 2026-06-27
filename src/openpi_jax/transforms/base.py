"""Composable data transforms.

A ``DataTransform`` maps a dict -> dict. They are chained with ``Compose`` and applied in the data pipeline
(input transforms) and at inference (output transforms). This mirrors OpenPI's transform protocol.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np

from openpi_jax.models.tokenizer import PaligemmaTokenizer
from openpi_jax.shared.normalize import Normalizer


@runtime_checkable
class DataTransform(Protocol):
    def __call__(self, data: dict) -> dict: ...


@dataclasses.dataclass
class Compose:
    transforms: list[DataTransform]

    def __call__(self, data: dict) -> dict:
        for t in self.transforms:
            data = t(data)
        return data


@dataclasses.dataclass
class Normalize:
    normalizer: Normalizer

    def __call__(self, data: dict) -> dict:
        return self.normalizer.normalize(data)


@dataclasses.dataclass
class Unnormalize:
    normalizer: Normalizer

    def __call__(self, data: dict) -> dict:
        return self.normalizer.unnormalize(data)


@dataclasses.dataclass
class TokenizePrompt:
    tokenizer: PaligemmaTokenizer
    prompt_key: str = "prompt"

    def __call__(self, data: dict) -> dict:
        prompt = data.get(self.prompt_key, "")
        ids, mask = self.tokenizer.encode(prompt)
        out = dict(data)
        out["tokenized_prompt"] = np.asarray(ids)
        out["tokenized_prompt_mask"] = np.asarray(mask)
        return out
