#!/bin/bash
# Run throughput benchmarks across model configs
set -euo pipefail

echo "=== Toy model ==="
python src/minimaker/benchmark.py model=toy data=toy training=debug

echo ""
echo "=== Small model ==="
python src/minimaker/benchmark.py model=small

echo ""
echo "=== Medium model ==="
python src/minimaker/benchmark.py model=medium
