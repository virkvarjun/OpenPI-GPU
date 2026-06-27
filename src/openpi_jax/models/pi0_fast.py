"""π0-FAST: autoregressive VLA over FAST-tokenized actions.

Instead of flow matching, π0-FAST tokenizes action chunks with the FAST tokenizer (see
``tokenizer.FastActionTokenizer``) and decodes them autoregressively with the Gemma backbone, exactly like text.
This makes training simpler (cross-entropy) at the cost of slower inference.

Scaffold only — shares the SigLIP + Gemma backbone with π0; the trunk and KV-cache decode loop are TODOs.
"""

from __future__ import annotations

import dataclasses

import flax.linen as nn

from openpi_jax.models.gemma import GEMMA_2B, GemmaConfig
from openpi_jax.models.observation import Observation
from openpi_jax.models.siglip import SiglipConfig
from openpi_jax.shared.typing import Array, PRNGKey


@dataclasses.dataclass(frozen=True)
class Pi0FastConfig:
    action_dim: int = 7
    action_horizon: int = 50
    max_token_len: int = 256
    fast_vocab_size: int = 1024
    vlm: GemmaConfig = GEMMA_2B
    vision: SiglipConfig = SiglipConfig()
    dtype: str = "bfloat16"


class Pi0Fast(nn.Module):
    config: Pi0FastConfig

    @nn.compact
    def __call__(self, obs: Observation, target_tokens: Array) -> Array:
        """Next-token cross-entropy over FAST action tokens.

        TODO(modeling): embed prefix (images+text+state), append target tokens, run the causal Gemma trunk,
        and return mean cross-entropy on the action-token positions.
        """
        raise NotImplementedError("π0-FAST training step — see docs/ROADMAP.md")

    def sample_actions(self, rng: PRNGKey, obs: Observation) -> Array:
        """Greedy/temperature autoregressive decode of FAST tokens, then de-tokenize to actions."""
        raise NotImplementedError("π0-FAST autoregressive decode — see docs/ROADMAP.md")
