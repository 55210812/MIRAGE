#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/runs/data1-deepseek-mirage"
BEGIN="# mirage-data1-deepseek-watchdog-BEGIN"
END="# mirage-data1-deepseek-watchdog-END"
BLOCK="$(cat <<CRON
$BEGIN
@reboot /bin/bash -lc 'sleep 120; cd $ROOT_DIR && bash scripts/data1_deepseek_watchdog.sh >> runs/data1-deepseek-mirage/watchdog.log 2>&1'
*/10 * * * * /bin/bash -lc 'cd $ROOT_DIR && bash scripts/data1_deepseek_watchdog.sh >> runs/data1-deepseek-mirage/watchdog.log 2>&1'
$END
CRON
)"

mkdir -p "$RUN_DIR"
tmp_current="$(mktemp)"
tmp_next="$(mktemp)"
trap 'rm -f "$tmp_current" "$tmp_next"' EXIT
crontab -l > "$tmp_current" 2>/dev/null || true
awk -v begin="$BEGIN" -v end="$END" '
  $0 == begin {skip=1; next}
  $0 == end {skip=0; next}
  skip != 1 {print}
' "$tmp_current" > "$tmp_next"
{
  sed -n '$p' "$tmp_next" | grep -q '^$' || true
  cat "$tmp_next"
  if [[ -s "$tmp_next" ]]; then
    echo
  fi
  printf "%s\n" "$BLOCK"
} | crontab -
echo "installed crontab block: mirage-data1-deepseek-watchdog"
