#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-fy3ggpm_training}"
CONFIG_PATH="${2:-$ROOT_DIR/configs/default.yaml}"
DATA_ROOT="${3:-}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
    "bash '$ROOT_DIR/scripts/run_gpm_training.sh' '$CONFIG_PATH' '$DATA_ROOT'; bash"

echo "Started training in tmux session '$SESSION_NAME'. Attach with: tmux attach -t $SESSION_NAME"
