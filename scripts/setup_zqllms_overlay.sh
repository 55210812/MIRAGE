#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/hqdeng7/.conda/envs/zqllms/bin/python}"
OVERLAY="${OVERLAY:-$ROOT_DIR/.envs/zqllms-overlay}"

with_proxy_fallback() {
  "$@" && return 0

  echo "[overlay] direct command failed; retrying through remote 127.0.0.1:7890" >&2
  (
    export http_proxy="http://127.0.0.1:7890"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    "$@"
  ) && return 0

  echo "[overlay] remote 7890 failed; retrying through SSH reverse proxy 127.0.0.1:17890 if present" >&2
  (
    export http_proxy="http://127.0.0.1:17890"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    "$@"
  )
}

if [[ ! -x "$PYTHON" ]]; then
  echo "[overlay] missing Python: $PYTHON" >&2
  exit 1
fi

mkdir -p "$OVERLAY"
with_proxy_fallback "$PYTHON" -m pip install \
  --target "$OVERLAY" \
  --no-cache-dir \
  --no-deps \
  inseq==0.7.1 \
  captum==0.9.0 \
  nltk==3.9.4 \
  jsonlines==4.0.0 \
  jaxtyping==0.3.10 \
  typeguard==4.5.2 \
  rouge-score==0.1.2 \
  absl-py==2.3.1 \
  wadler-lindig==0.1.7 \
  treescope==0.1.10

PYTHONPATH="$OVERLAY" "$PYTHON" - <<'PY'
import torch
import transformers
import inseq
import nltk
import jsonlines
import captum
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("inseq", getattr(inseq, "__version__", "unknown"))
print("nltk", nltk.__version__)
PY
