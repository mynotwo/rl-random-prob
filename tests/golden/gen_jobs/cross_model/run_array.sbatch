#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --array=0-7%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=1:30:00
# smollm2_olmo3_supplement_20260920 / frozen: 8 jobs, one GPU each. Finished jobs are skipped.
set -uo pipefail
export HF_HOME=${PROBE_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}} TOKENIZERS_PARALLELISM=false
cd <REPO>/src
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" <OUT>/jobs.jsonl)
out=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["out"])' "$line")
marker=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["done_marker"])' "$line")
cmd=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["cmd"])' "$line")
if [[ -f $out ]] && grep -q "$marker" "$out"; then echo "skip: $out already complete"; exit 0; fi
mkdir -p "$(dirname "$out")"
sleep $((RANDOM % 1))   # stagger: many tasks cold-starting CUDA on one node at once hit context-creation OOM (2026-09-17)
for attempt in 1 2; do
  t0=$(date +%s); echo "[$(date -Is)] start idx=$SLURM_ARRAY_TASK_ID attempt=$attempt host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  python3 -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
  echo "$cmd"
  eval "$cmd" 2> >(tee /tmp/exp_err_$SLURM_JOB_ID >&2); rc=$?
  echo "{\"idx\": $SLURM_ARRAY_TASK_ID, \"attempt\": $attempt, \"job\": \"$SLURM_JOB_ID\", \"host\": \"$(hostname)\", \"start\": $t0, \"end\": $(date +%s), \"rc\": $rc, \"out\": \"$out\"}" >> <OUT>/run_status.jsonl
  # retry once only for the infrastructure failure seen on 2026-09-17: CUDA OOM while creating the context
  if [[ $rc -ne 0 && $attempt -eq 1 ]] && grep -q "out of memory" /tmp/exp_err_$SLURM_JOB_ID && ! grep -qs '"phase": "train"' "$out"; then echo "load-time CUDA OOM, retrying once in 60 s"; rm -f "$out"; sleep 60; continue; fi
  break
done
rm -f /tmp/exp_err_$SLURM_JOB_ID
echo "[$(date -Is)] done rc=$rc"; exit $rc
