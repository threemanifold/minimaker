#!/bin/bash
# Launch training — single GPU or multi-GPU via torchrun
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training on $NUM_GPUS GPUs"
    # Use SHM transport for single-node; avoids missing NCCL NET plugin on GCP
    export NCCL_NET=Socket
    torchrun --nproc_per_node="$NUM_GPUS" src/minimaker/train.py "$@"
else
    echo "Launching single-device training"
    python src/minimaker/train.py "$@"
fi
