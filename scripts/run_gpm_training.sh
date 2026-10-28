#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/default.yaml}"
DATA_ROOT="${2:-$(PYTHONPATH="$ROOT_DIR" python3 -c 'import sys; from utils.config import load_config; print(load_config(sys.argv[1]).data.dataset_root)' "$CONFIG_PATH")}"
OUTPUT_DIR="${3:-}"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(PYTHONPATH="$ROOT_DIR" python3 -c 'import sys; from utils.config import load_config; print(load_config(sys.argv[1]).train.output_dir)' "$CONFIG_PATH")"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting FY3G-GPM training"
echo "Config: $CONFIG_PATH"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

python3 "$ROOT_DIR/scripts/train.py" \
    --config "$CONFIG_PATH" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$OUTPUT_DIR/console.log"
