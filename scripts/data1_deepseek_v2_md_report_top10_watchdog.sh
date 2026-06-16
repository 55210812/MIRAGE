#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/data1-deepseek-mirage-v2-md-report-top10}"
export SESSION="${SESSION:-mirage-md-report-top10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HISTORY_SOURCE="${HISTORY_SOURCE:-md_report}"
export HISTORY_FILE_NAME="${HISTORY_FILE_NAME:-md_report.txt}"
export CTI_MODE="${CTI_MODE:-sentence_logprob}"
export DOC_LIMIT="${DOC_LIMIT:-100}"
export WORKDIR_LIMIT="${WORKDIR_LIMIT:-1}"
export TOP_SENSITIVE_PERCENT="${TOP_SENSITIVE_PERCENT:-0.10}"
export TOP_DOC_PERCENT="${TOP_DOC_PERCENT:-0.10}"

bash "$ROOT_DIR/scripts/data1_deepseek_v2_watchdog.sh"
