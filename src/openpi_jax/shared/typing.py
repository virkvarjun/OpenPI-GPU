"""Lightweight type aliases used across the codebase."""

from __future__ import annotations

from typing import Any

import jax

Array = jax.Array
PyTree = Any
PRNGKey = jax.Array

# A batch is a nested dict of arrays. We keep it loose here and validate at the model boundary.
Batch = dict[str, Any]
Params = PyTree
