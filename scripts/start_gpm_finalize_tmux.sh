#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-fy3ggpm_finalize}"
CONFIG_PATH="${2:-$ROOT_DIR/configs/default.yaml}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
    "bash '$ROOT_DIR/scripts/run_gpm_finalize.sh' '$CONFIG_PATH'; bash"

echo "Started finalization in tmux session '$SESSION_NAME'. Attach with: tmux attach -t $SESSION_NAME"
