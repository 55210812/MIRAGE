#!/usr/bin/env bash
set -Eeuo pipefail

SESSION="${SESSION:-mirage-data1-deepseek-v2}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "stopped $SESSION"
else
  echo "$SESSION is not running"
fi
