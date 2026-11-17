#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --array=0-218%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=4:00:00
# checkpoint_curve_completion_20260915: 219 jobs ({'random': 104, 'ground_truth': 100, 'frozen': 13, 'frozen_topup': 2}). One GPU each; finished jobs are skipped; re-submit failed ids with --array=<ids>.
set -uo pipefail
export HF_HOME=${PROBE_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}} TOKENIZERS_PARALLELISM=false
cd <REPO>/src
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" <OUT>/jobs.jsonl)
out=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["out"])' "$line")
marker=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["done_marker"])' "$line")
cmd=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["cmd"])' "$line")
if [[ -f $out ]] && grep -q "$marker" "$out"; then echo "skip: $out already complete"; exit 0; fi
mkdir -p "$(dirname "$out")"
attempt=$(( $(grep -c "\"idx\": $SLURM_ARRAY_TASK_ID," <OUT>/run_status.jsonl 2>/dev/null) + 1 ))
t0=$(date +%s); echo "[$(date -Is)] start idx=$SLURM_ARRAY_TASK_ID attempt=$attempt host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "$cmd"
eval "$cmd"; rc=$?
echo "{\"idx\": $SLURM_ARRAY_TASK_ID, \"attempt\": $attempt, \"job\": \"$SLURM_JOB_ID\", \"host\": \"$(hostname)\", \"start\": $t0, \"end\": $(date +%s), \"rc\": $rc, \"out\": \"$out\"}" >> <OUT>/run_status.jsonl
echo "[$(date -Is)] done rc=$rc"; exit $rc
