#!/bin/bash
# Launch DPO training — single GPU or multi-GPU via torchrun
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed DPO on $NUM_GPUS GPUs"
    torchrun --nproc_per_node="$NUM_GPUS" src/minimaker/dpo.py "$@"
else
    echo "Launching single-device DPO"
    python src/minimaker/dpo.py "$@"
fi
