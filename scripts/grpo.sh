#!/bin/bash
# Launch GRPO training — single GPU or multi-GPU via torchrun
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed GRPO on $NUM_GPUS GPUs"
    torchrun --nproc_per_node="$NUM_GPUS" src/minimaker/grpo.py "$@"
else
    echo "Launching single-device GRPO"
    python src/minimaker/grpo.py "$@"
fi
