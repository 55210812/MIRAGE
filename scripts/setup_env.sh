#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$ROOT_DIR/.envs/mirage-py39}"
CONDA_BIN="${CONDA_BIN:-}"

with_proxy_fallback() {
  "$@" && return 0

  echo "[setup] direct command failed; retrying through remote 127.0.0.1:7890" >&2
  (
    export http_proxy="http://127.0.0.1:7890"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    "$@"
  ) && return 0

  echo "[setup] remote 7890 failed; retrying through SSH reverse proxy 127.0.0.1:17890 if present" >&2
  (
    export http_proxy="http://127.0.0.1:17890"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    "$@"
  )
}

find_conda() {
  if [[ -n "$CONDA_BIN" ]]; then
    printf '%s\n' "$CONDA_BIN"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/mnt/data2/hyx111/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  echo "[setup] environment already exists: $ENV_PREFIX"
  "$ENV_PREFIX/bin/python" -V
  exit 0
fi

CONDA="$(find_conda)" || {
  echo "[setup] conda not found. Set CONDA_BIN=/path/to/conda and rerun." >&2
  exit 1
}

TMP_ENV="$(mktemp)"
trap 'rm -f "$TMP_ENV"' EXIT
grep -v '^prefix:' "$ROOT_DIR/MIRAGE.yaml" > "$TMP_ENV"

mkdir -p "$(dirname "$ENV_PREFIX")"
with_proxy_fallback "$CONDA" env create -p "$ENV_PREFIX" -f "$TMP_ENV"
"$ENV_PREFIX/bin/python" -V
