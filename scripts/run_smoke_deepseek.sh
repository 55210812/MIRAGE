#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$ROOT_DIR/.envs/mirage-py39}"
OVERLAY="${OVERLAY:-$ROOT_DIR/.envs/zqllms-overlay}"
if [[ -z "${PYTHON:-}" && -x /home/hqdeng7/.conda/envs/zqllms/bin/python && -d "$OVERLAY" ]]; then
  PYTHON="/home/hqdeng7/.conda/envs/zqllms/bin/python"
elif [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$ENV_PREFIX/bin/python"
fi
CONFIG="${CONFIG:-configs/eli5_deepseek_qwen14b_shot0_ndoc2_bm25_selfcitation_smoke.yaml}"
OUT="result/selfcitation/eli5-DeepSeek-R1-Distill-Qwen-14B-bm25-smoke-shot0-ndoc2-42-quick_test1.json"
INTERNAL="internal_selfcitation/_home_intern_models_deepseek-r1-distill-qwen-14b-shot0-seed42-0.json"

if [[ -d "$OVERLAY" ]]; then
  export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
fi
export NLTK_DATA="${NLTK_DATA:-$ROOT_DIR/.nltk_data}"

with_proxy_fallback() {
  "$@" && return 0

  echo "[smoke] direct command failed; retrying through remote 127.0.0.1:7890" >&2
  (
    export http_proxy="http://127.0.0.1:7890"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    "$@"
  ) && return 0

  echo "[smoke] remote 7890 failed; retrying through SSH reverse proxy 127.0.0.1:17890 if present" >&2
  (
    export http_proxy="http://127.0.0.1:17890"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    "$@"
  )
}

cd "$ROOT_DIR"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "[smoke] missing Python environment: $PYTHON" >&2
  echo "[smoke] run scripts/setup_env.sh first, install the zqllms overlay, or set PYTHON=/path/to/python" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import torch
import transformers
import inseq
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("inseq", getattr(inseq, "__version__", "unknown"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY

test -d /home/intern/models/DeepSeek-R1-Distill-Qwen-14B

cd "$ROOT_DIR/sec5_longQA"

if [[ ! -f data/eli5_eval_bm25_top100.json && ! -f smoke_data/eli5_eval_bm25_top100_smoke.json ]]; then
  with_proxy_fallback bash 0_download_data.sh
fi

if [[ ! -s "$OUT" || "${FORCE_GENERATE:-0}" == "1" ]]; then
  with_proxy_fallback "$PYTHON" run.py --config "$CONFIG"
else
  echo "[smoke] reusing existing generation output: $OUT"
fi
test -s "$OUT"

if [[ ! -s "$INTERNAL" || "${FORCE_ATTRIBUTION:-0}" == "1" ]]; then
  "$PYTHON" mirage_attribute.py --f "$OUT"
else
  echo "[smoke] reusing existing attribution output: $INTERNAL"
fi
test -s "$INTERNAL"

if [[ ! -s "$OUT.mirage_cite_CTI_1_CCI_-5" || "${FORCE_CITE:-0}" == "1" ]]; then
  "$PYTHON" mirage_cite.py --f "$OUT" --CTI 1 --CCI -5
else
  echo "[smoke] reusing existing citation output: $OUT.mirage_cite_CTI_1_CCI_-5"
fi
test -s "$OUT.mirage_cite_CTI_1_CCI_-5"

"$PYTHON" eval.py --f "$OUT.mirage_cite_CTI_1_CCI_-5" --citations --citation_regex_only
test -s "$OUT.mirage_cite_CTI_1_CCI_-5.score"

echo "[smoke] MIRAGE DeepSeek Qwen 14B smoke test passed"
