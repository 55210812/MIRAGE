#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/data1-deepseek-mirage-v2}"
SESSION="${SESSION:-mirage-data1-deepseek-v2}"

echo "root: $ROOT_DIR"
echo "run_dir: $RUN_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux: running ($SESSION)"
else
  echo "tmux: not running ($SESSION)"
fi

for marker in .running .failed .done .heartbeat; do
  if [[ -f "$RUN_DIR/$marker" ]]; then
    echo
    echo "== $marker =="
    sed -n '1,60p' "$RUN_DIR/$marker"
  fi
done

echo
echo "== counts =="
for file in manifest.json cti_failed.jsonl sentence_cti.jsonl doc_perturbation.jsonl paragraph_perturbation.jsonl summary.md; do
  if [[ -f "$RUN_DIR/$file" ]]; then
    if [[ "$file" == *.jsonl ]]; then
      printf "%-32s %s lines\n" "$file" "$(wc -l < "$RUN_DIR/$file" | tr -d ' ')"
    else
      printf "%-32s present\n" "$file"
    fi
  else
    printf "%-32s missing\n" "$file"
  fi
done
if [[ -d "$RUN_DIR/internal_cti" ]]; then
  cti_cache_count="$(find "$RUN_DIR/internal_cti" -type f -name 'sentence-*.json' ! -name '*.inseq.json' | wc -l | tr -d ' ')"
  printf "%-32s %s files\n" "internal_cti cache" "$cti_cache_count"
fi
if [[ -d "$RUN_DIR/generated_history" ]]; then
  printf "%-32s %s files\n" "generated_history" "$(find "$RUN_DIR/generated_history" -type f -name '历史成果.txt' | wc -l | tr -d ' ')"
fi

echo
echo "== recent run.log =="
if [[ -f "$RUN_DIR/run.log" ]]; then
  tail -60 "$RUN_DIR/run.log"
else
  echo "missing"
fi

echo
echo "== recent watchdog.log =="
if [[ -f "$RUN_DIR/watchdog.log" ]]; then
  tail -20 "$RUN_DIR/watchdog.log"
else
  echo "missing"
fi

echo
echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits || true
else
  echo "nvidia-smi missing"
fi
