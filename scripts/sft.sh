#!/bin/bash
# Launch SFT — single GPU or multi-GPU via torchrun
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed SFT on $NUM_GPUS GPUs"
    torchrun --nproc_per_node="$NUM_GPUS" src/minimaker/sft.py "$@"
else
    echo "Launching single-device SFT"
    python src/minimaker/sft.py "$@"
fi
