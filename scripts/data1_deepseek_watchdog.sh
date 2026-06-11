#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/data1-deepseek-mirage}"
SESSION="${SESSION:-mirage-data1-deepseek}"
LOCK_FILE="${LOCK_FILE:-/tmp/${SESSION}.lock}"

mkdir -p "$RUN_DIR"

(
  flock -n 9 || {
    echo "[$(date -Is)] watchdog already running"
    exit 0
  }

  cd "$ROOT_DIR"
  if [[ -f "$RUN_DIR/.done" ]]; then
    echo "[$(date -Is)] done marker exists; not starting"
    exit 0
  fi
  if [[ -f "$RUN_DIR/.failed" ]]; then
    echo "[$(date -Is)] failed marker exists; not starting"
    exit 0
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[$(date -Is)] session $SESSION already running"
    exit 0
  fi

  echo "[$(date -Is)] starting tmux session $SESSION"
  tmux new-session -d -s "$SESSION" \
    "cd '$ROOT_DIR' && bash scripts/run_data1_deepseek_attribution.sh >> '$RUN_DIR/run.log' 2>&1"
) 9>"$LOCK_FILE"
