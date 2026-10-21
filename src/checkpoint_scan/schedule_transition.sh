#!/usr/bin/env bash
# --- portability shim (added when consolidating 138/141) -------------------
# Override any of these to run outside the original hosts.
REPO_ROOT="${PROBE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROBE_PYTHON="${PROBE_PYTHON:-python3}"
PROBE_OUT="${PROBE_OUT:-$REPO_ROOT/experiments}"
PROBE_HF_HOME="${PROBE_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
# --------------------------------------------------------------------------
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 START_EPOCH DEADLINE_EPOCH OUTPUT_ROOT GPU_LIST TAG_LIST LOG" >&2
  exit 2
fi
start_epoch=$1
deadline_epoch=$2
output_root=$3
gpu_list=$4
tag_list=$5
schedule_log=$6
launcher=${REPO_ROOT}/src/checkpoint_scan/launch_workers.sh

while (( $(date +%s) < start_epoch )); do
  remaining=$((start_epoch - $(date +%s)))
  (( remaining > 60 )) && sleep 60 || sleep "$remaining"
done

{
  echo "transition_started=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "deadline_epoch=$deadline_epoch"
  bash "$launcher" "$output_root" "$deadline_epoch" "$gpu_list" "$tag_list"
} >> "$schedule_log" 2>&1
