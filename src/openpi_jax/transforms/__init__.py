"""Robot-/dataset-specific input & output transforms.

Each robot embodiment maps its raw observation (camera names, joint layout) into the model's canonical
observation, and maps the model's action chunk back to robot commands. These are kept separate from the model so
the same π0 can be reused across embodiments by swapping transforms.
"""

from openpi_jax.transforms.base import (
    Compose,
    DataTransform,
    Normalize,
    TokenizePrompt,
    Unnormalize,
)

__all__ = ["DataTransform", "Compose", "Normalize", "Unnormalize", "TokenizePrompt"]
