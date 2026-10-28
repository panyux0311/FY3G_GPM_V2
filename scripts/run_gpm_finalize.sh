#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/default.yaml}"
OUTPUT_DIR="${2:-}"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(PYTHONPATH="$ROOT_DIR" python3 -c 'import sys; from utils.config import load_config; print(load_config(sys.argv[1]).train.output_dir)' "$CONFIG_PATH")"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting FY3G-GPM finalization"
python3 "$ROOT_DIR/scripts/finalize.py" \
    --config "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR"
