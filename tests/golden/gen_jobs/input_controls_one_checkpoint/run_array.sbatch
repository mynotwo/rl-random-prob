#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --array=0-3%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=3:00:00
# random question + random reward @ p2_50B: 4 runs (arms C/A x 2 seeds). Finished runs are skipped.
set -uo pipefail
export HF_HOME=${PROBE_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}} TOKENIZERS_PARALLELISM=false
cd <REPO>/src
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" <OUT>/jobs.jsonl)
out=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["out"])' "$line")
cmd=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["cmd"])' "$line")
if [[ -f $out ]] && grep -q '"phase": "after"' "$out"; then echo "skip: $out already complete"; exit 0; fi
mkdir -p "$(dirname "$out")"
sleep $((RANDOM % 45))   # stagger: many tasks cold-starting CUDA on one idle node at once hit context-creation OOM (2026-09-17)
for attempt in 1 2; do
  t0=$(date +%s); echo "[$(date -Is)] start idx=$SLURM_ARRAY_TASK_ID attempt=$attempt host=$(hostname)"
  echo "$cmd"
  eval "$cmd" 2> >(tee /tmp/exp_err_$SLURM_JOB_ID >&2); rc=$?
  echo "{\"idx\": $SLURM_ARRAY_TASK_ID, \"attempt\": $attempt, \"job\": \"$SLURM_JOB_ID\", \"host\": \"$(hostname)\", \"start\": $t0, \"end\": $(date +%s), \"rc\": $rc, \"out\": \"$out\"}" >> <OUT>/run_status.jsonl
  # retry once only for the infrastructure failure: CUDA OOM before any training record exists
  if [[ $rc -ne 0 && $attempt -eq 1 ]] && grep -q "out of memory" /tmp/exp_err_$SLURM_JOB_ID && ! grep -qs '"phase": "train"' "$out"; then
    echo "load-time CUDA OOM, retrying once in 60 s"; rm -f "$out"; sleep 60; continue
  fi
  break
done
rm -f /tmp/exp_err_$SLURM_JOB_ID
echo "[$(date -Is)] done rc=$rc"; exit $rc
