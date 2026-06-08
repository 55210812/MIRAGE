#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$ROOT_DIR/.envs/mirage-py39}"
PYTHON="${PYTHON:-$ENV_PREFIX/bin/python}"

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
  echo "[smoke] run scripts/setup_env.sh first, or set PYTHON=/path/to/python" >&2
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

cd "$ROOT_DIR/sec5_longQA"

if [[ ! -f data/eli5_eval_bm25_top100.json ]]; then
  with_proxy_fallback bash 0_download_data.sh
fi

with_proxy_fallback "$PYTHON" run.py --config configs/eli5_llama32_3b_shot0_ndoc2_bm25_selfcitation_smoke.yaml

OUT="result/selfcitation/eli5-Llama-3.2-3B-Instruct-bm25-smoke-shot0-ndoc2-42-quick_test2.json"
test -s "$OUT"

with_proxy_fallback "$PYTHON" mirage_attribute.py --f "$OUT"
test -n "$(find internal_selfcitation -maxdepth 1 -type f -name 'meta-llama_llama-3.2-3b-instruct-shot0-seed42-*.json' -print -quit)"

"$PYTHON" mirage_cite.py --f "$OUT" --CTI 1 --CCI -5
test -s "$OUT.mirage_cite_CTI_1_CCI_-5"

"$PYTHON" eval.py --f "$OUT.mirage_cite_CTI_1_CCI_-5" --citations
test -s "$OUT.mirage_cite_CTI_1_CCI_-5.score"

echo "[smoke] MIRAGE Llama 3.2 3B smoke test passed"
