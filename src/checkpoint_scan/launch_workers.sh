#!/usr/bin/env bash
# --- portability shim (added when consolidating 138/141) -------------------
# Override any of these to run outside the original hosts.
REPO_ROOT="${PROBE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROBE_PYTHON="${PROBE_PYTHON:-python3}"
PROBE_OUT="${PROBE_OUT:-$REPO_ROOT/experiments}"
PROBE_HF_HOME="${PROBE_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
# --------------------------------------------------------------------------
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 OUTPUT_ROOT DEADLINE_EPOCH GPU_LIST TAG_LIST" >&2
  exit 2
fi
output_root=$1
deadline=$2
gpu_list=$3
tag_list=$4
scan_dir=${REPO_ROOT}/src/checkpoint_scan
python_bin=${PROBE_PYTHON}
proxy_url=${HTTP_PROXY:-}   # set HTTP_PROXY in the environment if your site needs a proxy
hf_home=${PROBE_HF_HOME}
if [[ ! -d "$hf_home" ]]; then
  hf_home=${PROBE_HF_HOME}_checkpoint_scan
  mkdir -p "$hf_home"
fi
mkdir -p "$output_root/worker_logs"
for gpu in ${gpu_list//,/ }; do
  worker="$(hostname)-gpu${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    HTTP_PROXY="$proxy_url" HTTPS_PROXY="$proxy_url" \
    http_proxy="$proxy_url" https_proxy="$proxy_url" HF_HOME="$hf_home" \
    nohup "$python_bin" "$scan_dir/worker.py" --scan_dir "$scan_dir" \
      --output_root "$output_root" --deadline "$deadline" --worker "$worker" \
      --tags "$tag_list" > "$output_root/worker_logs/$worker.log" 2>&1 < /dev/null &
  echo "$! $worker"
done
