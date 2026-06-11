#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/data1-deepseek-mirage}"
ENV_PREFIX="${ENV_PREFIX:-$ROOT_DIR/.envs/mirage-py39}"
OVERLAY="${OVERLAY:-$ROOT_DIR/.envs/zqllms-overlay}"
if [[ -z "${PYTHON:-}" && -x /home/hqdeng7/.conda/envs/zqllms/bin/python && -d "$OVERLAY" ]]; then
  PYTHON="/home/hqdeng7/.conda/envs/zqllms/bin/python"
elif [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$ENV_PREFIX/bin/python"
fi

mkdir -p "$RUN_DIR"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -d "$OVERLAY" ]]; then
  export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
fi
export NLTK_DATA="${NLTK_DATA:-$ROOT_DIR/.nltk_data}"

if [[ ! -x "$PYTHON" ]]; then
  echo "[data1] missing Python environment: $PYTHON" >&2
  exit 1
fi

if [[ -f "$RUN_DIR/.done" && "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "[data1] .done exists; set FORCE_RERUN=1 to run again"
  exit 0
fi

on_error() {
  local exit_code=$?
  {
    echo "failed_at=$(date -Is)"
    echo "exit_code=$exit_code"
  } > "$RUN_DIR/.failed"
  rm -f "$RUN_DIR/.running"
  exit "$exit_code"
}
trap on_error ERR

rm -f "$RUN_DIR/.failed"
echo "started_at=$(date -Is)" > "$RUN_DIR/.running"
echo "[data1] started_at=$(date -Is)"
echo "[data1] python=$PYTHON"
echo "[data1] run_dir=$RUN_DIR"
echo "[data1] workdir_limit=${WORKDIR_LIMIT:-1} doc_limit=${DOC_LIMIT:-100} cti_method=${CTI_METHOD:-saliency} cti_doc_chars=${CTI_DOC_CHARS:-0}"

"$PYTHON" "$ROOT_DIR/sec5_longQA/data1_deepseek_attribution.py" \
  --data-root "${DATA1_ROOT:-$ROOT_DIR/data_1}" \
  --output-dir "$RUN_DIR" \
  --model "${DATA1_MODEL:-/home/intern/models/DeepSeek-R1-Distill-Qwen-14B}" \
  --workdir-limit "${WORKDIR_LIMIT:-1}" \
  --doc-limit "${DOC_LIMIT:-100}" \
  --cti-context-docs "${CTI_CONTEXT_DOCS:-5}" \
  --cti-doc-chars "${CTI_DOC_CHARS:-0}" \
  --cti-method "${CTI_METHOD:-saliency}" \
  --top-sensitive-sentences "${TOP_SENSITIVE_SENTENCES:-100}" \
  --paragraph-doc-topk "${PARAGRAPH_DOC_TOPK:-3}" \
  "$@"

echo "done_at=$(date -Is)" > "$RUN_DIR/.done"
rm -f "$RUN_DIR/.running"
