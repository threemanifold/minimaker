#!/bin/bash
# Download + tokenize dataset. No GPU needed.
# Usage: bash scripts/prepare_data.sh data=openwebtext
set -euo pipefail

python -m minimaker.data "$@"
