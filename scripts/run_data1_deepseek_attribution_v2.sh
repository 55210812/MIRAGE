#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/data1-deepseek-mirage-v2}"
ENV_PREFIX="${ENV_PREFIX:-$ROOT_DIR/.envs/mirage-py39}"
OVERLAY="${OVERLAY:-$ROOT_DIR/.envs/zqllms-overlay}"
if [[ -z "${PYTHON:-}" && -x /home/hqdeng7/.conda/envs/zqllms/bin/python && -d "$OVERLAY" ]]; then
  PYTHON="/home/hqdeng7/.conda/envs/zqllms/bin/python"
elif [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$ENV_PREFIX/bin/python"
fi

with_proxy_fallback() {
  "$@" && return 0
  echo "[data1-v2] direct command failed; retrying through remote 127.0.0.1:7890" >&2
  (
    export http_proxy="http://127.0.0.1:7890"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    "$@"
  ) && return 0
  echo "[data1-v2] remote 7890 failed; retrying through SSH reverse proxy 127.0.0.1:17890 if present" >&2
  (
    export http_proxy="http://127.0.0.1:17890"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    "$@"
  )
}

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
  echo "[data1-v2] missing Python environment: $PYTHON" >&2
  exit 1
fi

EMBEDDING_MODEL_RESOLVED="${EMBEDDING_MODEL:-BAAI/bge-m3}"
if [[ "$EMBEDDING_MODEL_RESOLVED" == "BAAI/bge-m3" ]]; then
  BGE_SNAPSHOT="$("$PYTHON" - <<'PY' || true
from pathlib import Path
home = Path.home()
root = home / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3" / "snapshots"
for path in sorted(root.glob("*")) if root.exists() else []:
    if (path / "config.json").exists() and (path / "tokenizer_config.json").exists() and (path / "pytorch_model.bin").exists():
        print(path)
        break
PY
)"
  if [[ -n "$BGE_SNAPSHOT" ]]; then
    EMBEDDING_MODEL_RESOLVED="$BGE_SNAPSHOT"
    echo "[data1-v2] using cached BAAI/bge-m3 snapshot: $EMBEDDING_MODEL_RESOLVED"
  fi
fi

if [[ -f "$RUN_DIR/.done" && "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "[data1-v2] .done exists; set FORCE_RERUN=1 to run again"
  exit 0
fi

if [[ "${SKIP_EMBEDDING_PREFETCH:-0}" != "1" ]]; then
  with_proxy_fallback "$PYTHON" -c "from transformers import AutoTokenizer, AutoModel; m='${EMBEDDING_MODEL_RESOLVED}'; AutoTokenizer.from_pretrained(m, trust_remote_code=True); AutoModel.from_pretrained(m, trust_remote_code=True)"
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
echo "[data1-v2] started_at=$(date -Is)"
echo "[data1-v2] python=$PYTHON"
echo "[data1-v2] run_dir=$RUN_DIR"
echo "[data1-v2] workdir_limit=${WORKDIR_LIMIT:-1} doc_limit=${DOC_LIMIT:-100} embedding=$EMBEDDING_MODEL_RESOLVED cti_mode=${CTI_MODE:-token_saliency} history_source=${HISTORY_SOURCE:-model_generate}"

"$PYTHON" "$ROOT_DIR/sec5_longQA/data1_deepseek_attribution_v2.py" \
  --data-root "${DATA1_ROOT:-$ROOT_DIR/data_1}" \
  --output-dir "$RUN_DIR" \
  --model "${DATA1_MODEL:-/home/intern/models/DeepSeek-R1-Distill-Qwen-14B}" \
  --embedding-model "$EMBEDDING_MODEL_RESOLVED" \
  --embedding-device "${EMBEDDING_DEVICE:-cpu}" \
  --workdir-limit "${WORKDIR_LIMIT:-1}" \
  --doc-limit "${DOC_LIMIT:-100}" \
  --cti-context-docs "${CTI_CONTEXT_DOCS:-5}" \
  --cti-mode "${CTI_MODE:-token_saliency}" \
  --history-source "${HISTORY_SOURCE:-model_generate}" \
  --history-file-name "${HISTORY_FILE_NAME:-md_report.txt}" \
  --top-sensitive-sentences "${TOP_SENSITIVE_SENTENCES:-100}" \
  --top-sensitive-percent "${TOP_SENSITIVE_PERCENT:-0}" \
  --paragraph-doc-topk "${PARAGRAPH_DOC_TOPK:-3}" \
  --top-doc-percent "${TOP_DOC_PERCENT:-0}" \
  --no-think-prefill \
  "$@"

echo "done_at=$(date -Is)" > "$RUN_DIR/.done"
rm -f "$RUN_DIR/.running"
