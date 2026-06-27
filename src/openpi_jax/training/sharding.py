"""Device mesh + sharding helpers for multi-accelerator training.

OpenPI shards parameters and data across a 2D mesh (FSDP over a data axis, optional tensor parallelism). This
module will hold the mesh construction and ``NamedSharding`` annotations. Stubbed for now.
"""

from __future__ import annotations

import jax


def make_mesh(data_parallel: int = -1, model_parallel: int = 1) -> jax.sharding.Mesh:
    """Build a (data, model) device mesh. ``-1`` means "use all remaining devices"."""
    devices = jax.devices()
    n = len(devices)
    if data_parallel == -1:
        data_parallel = n // model_parallel
    if data_parallel * model_parallel != n:
        raise ValueError(f"mesh {data_parallel}x{model_parallel} != {n} devices")
    mesh_devices = jax.experimental.mesh_utils.create_device_mesh((data_parallel, model_parallel))
    return jax.sharding.Mesh(mesh_devices, axis_names=("data", "model"))
