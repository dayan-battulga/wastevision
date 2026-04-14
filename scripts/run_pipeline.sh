#!/usr/bin/env bash
# End-to-end pipeline: download, convert, split, validate, then train.
#
# Usage: ./scripts/run_pipeline.sh [--skip-download]

set -euo pipefail

SKIP_DOWNLOAD=false
for arg in "$@"; do
  case $arg in
    --skip-download) SKIP_DOWNLOAD=true ;;
  esac
done

echo "=== DyrtyVision Pipeline ==="

if [ "$SKIP_DOWNLOAD" = false ]; then
  echo "[1/5] Downloading datasets..."
  python -m src.data.download
fi

echo "[2/5] Converting to YOLO format..."
# TODO: python -m src.data.convert

echo "[3/5] Splitting dataset..."
# TODO: python -m src.data.split

echo "[4/5] Validating annotations..."
# TODO: python -m src.data.validate

echo "[5/5] Training model..."
# TODO: python -m src.training.train --config configs/train.yaml

echo "=== Pipeline complete ==="
