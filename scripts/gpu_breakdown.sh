#!/bin/bash
# Run the V2 production step breakdown on a fresh GPU box (e.g. runpod).
#
# Usage (on the GPU host, or pipe over ssh):
#   bash scripts/gpu_breakdown.sh                 # if already inside the repo
#   curl -sL <raw-url>/scripts/gpu_breakdown.sh | bash   # self-bootstraps via git clone
#
# Profiles the REAL openpi pi0 step (random-init gemma_2b -- no checkpoint needed; attribution measures
# structure) and writes STEP_BREAKDOWN.md. Uses --optimizer none (forward+backward, the bottleneck region) so it
# fits a 32GB GPU; full AdamW state for ~2.5B params (fp32 m+v) would exceed that. A second SGD pass surfaces the
# optimizer category.
set +e
echo "REMOTE_RUN_START $(date -u +%H:%M:%S)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Bootstrap the repo if we're not already in it.
if [ ! -f scripts/profile_step_breakdown.py ]; then
  cd "${WORKDIR:-/root}" || exit 1
  rm -rf OpenPI-JAX
  git clone -b feat/jax-multihost-v1 --depth 1 https://github.com/virkvarjun/OpenPI-JAX.git 2>&1 | tail -2
  cd OpenPI-JAX || { echo "CLONE_FAILED"; exit 1; }
fi

pip install -q "jax[cuda12]==0.5.3" flax==0.10.2 optax einops "jaxtyping==0.2.36" \
  "beartype==0.19.0" "ml_collections==1.0.0" sentencepiece augmax equinox safetensors \
  transformers "numpy<2" pynvml 2>&1 | tail -3

export PYTHONPATH="$(pwd)/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # allocate on demand; avoids the 75% prealloc cap
python3 -c "import jax; print('JAX_DEVICES', jax.devices())" || { echo "JAX_IMPORT_FAILED"; exit 1; }

echo "=== gemma_2b forward+backward (optimizer=none) ==="
python3 scripts/profile_step_breakdown.py --variant gemma_2b --batch-size "${BATCH:-1}" --optimizer none --warmup 3 --iters 12
echo "REMOTE_RUN_COMPLETE $(date -u +%H:%M:%S)"
