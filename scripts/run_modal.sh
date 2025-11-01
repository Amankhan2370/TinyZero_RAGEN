#!/usr/bin/env bash
# Minimal runner intended for modal or other container-based runners.
# Usage: ./scripts/run_modal.sh [config_path]

set -euo pipefail

CONFIG=${1:-configs/default.yaml}
LOG_DIR=${LOG_DIR:-./logs_modal}

mkdir -p "$LOG_DIR"

echo "Starting training with config: $CONFIG"
python train.py --config "$CONFIG"
