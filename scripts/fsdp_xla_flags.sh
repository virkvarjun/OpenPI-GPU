# FSDP comms/compute-overlap XLA flags (source or export before launching training/profiling on GPU).
#
# Goal: hide FSDP's per-layer all-gather (unshard params) / reduce-scatter (shard grads) behind compute, so the
# weak-scaling curve flattens toward ideal (device time ~constant as GPUs grow → closer to linear aggregate
# throughput / ~35% MFU instead of the un-tuned 2.85× @ 24.8%).
#
# The levers:
#   - latency-hiding scheduler: let XLA reorder so collectives overlap independent compute.
#   - pipelined async collectives: issue layer N+1's all-gather while layer N's matmul runs.
#   - large combine thresholds: fuse many small collectives into fewer big ones that overlap better.
#   - while-loop double buffering: overlap across loop iterations (transformer layers).
#
# Usage:  source scripts/fsdp_xla_flags.sh
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/profile_fsdp.py --batch-size 16
#
# STATUS (2026-06-30): baseline reproduced (2-GPU default = 232 ms/step, 29.2% MFU); the tuned/4-GPU comparison
# had NOT completed when the H100 node was released — RE-RUN this before/after to quantify the overlap win.

export XLA_FLAGS="\
--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_all_gather_combine_threshold_bytes=1073741824 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824 \
--xla_gpu_enable_while_loop_double_buffering=true"
