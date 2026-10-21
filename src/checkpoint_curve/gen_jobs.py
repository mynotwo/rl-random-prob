#!/usr/bin/env python3
"""Expand configs/olmo2_checkpoints.json into the checkpoint-curve jobs (the 24h scan results
are reused): 13 frozen baselines, 104 Random + 100 GT 500-step runs, 2 high-precision
sample top-ups. Commands are the executed 24h argv verbatim (checkpoint_scan/worker.py), run with
the reference trainer trainer/rlvr_probe_v2_fixedsplit.py (sha 2a7e41...).

    python checkpoint_curve/gen_jobs.py --out runs/checkpoint_curve
Nothing is submitted; the script prints the sbatch line.
"""
from __future__ import annotations
import os

import argparse
import hashlib
import json
import pathlib
import shlex
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
MS = HERE.parent                      # src
REPO = MS.parent
MANIFEST = REPO / "configs" / "olmo2_checkpoints.json"
TRAINER = MS / "trainer" / "rlvr_probe_v2_fixedsplit.py"
FROZEN = MS / "trainer" / "frozen_eval.py"
DATASET = MS / "data" / "synthetic_gsm_rlvr_1000_seed13.jsonl"
MODEL = "allenai/OLMo-2-1124-7B"
PY = os.environ.get("PROBE_PYTHON", "python3")
PROTOCOL_ID = "24h_v2_fixedsplit_500"
KL = {"random": "0.01", "ground_truth": "0"}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def rl_job(ck, reward, seed, root):
    tag, rev = ck["tag"], ck["branch"]
    suffix = f"{tag}_{reward}_seed{seed}"
    cmd = [PY, str(TRAINER), "--dataset_path", str(DATASET), "--output_dir", f"{root}/{tag}",
           "--base_model", MODEL, "--revision", rev, "--train_mode", "original",
           "--reward_mode", reward, "--seed", str(seed), "--split_seed", "777",
           "--train_steps", "500", "--group_size", "4", "--max_new_tokens", "96",
           "--learning_rate", "3e-5", "--kl_coeff", KL[reward], "--eval_every", "100",
           "--eval_limit", "200", "--eval_batch_size", "8", "--lora_r", "8",
           "--lora_alpha", "16", "--lora_dropout", "0", "--run_suffix", suffix, "--save_adapter"]
    out = f"{root}/{tag}/rlvr_OLMo-2-1124-7B_{rev}_original_{reward}_seed{seed}_{suffix}.jsonl"
    key = dict(checkpoint_sha=ck["sha"], protocol_id=PROTOCOL_ID, reward_mode=reward, seed=seed, steps=500)
    return dict(kind="rl", tag=tag, revision=rev, key=key, out=out, done_marker='"phase": "after"', cmd=shlex.join(cmd))


def frozen_job(ck, root, samples=32, seed_start=41001, name="frozen"):
    tag, rev = ck["tag"], ck["branch"]
    out = f"{root}/{tag}/{name}.json"
    cmd = [PY, str(FROZEN), "--dataset_path", str(DATASET), "--output", out, "--base_model", MODEL,
           "--revision", rev, "--tag", tag, "--split_seed", "777", "--samples", str(samples),
           "--sample_seed_start", str(seed_start)]
    key = dict(checkpoint_sha=ck["sha"], protocol_id=PROTOCOL_ID, reward_mode=name, seed=seed_start, steps=samples)
    return dict(kind=name, tag=tag, revision=rev, key=key, out=out, done_marker='"summary"', cmd=shlex.join(cmd))


def interleave(cks):
    """Alternate early and late checkpoints so no wave is all one training stage."""
    cks = sorted(cks, key=lambda c: (c["stage"], c["step"]))
    out, lo, hi = [], 0, len(cks) - 1
    while lo <= hi:
        out.append(cks[lo]); lo += 1
        if lo <= hi:
            out.append(cks[hi]); hi -= 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    root = args.out.rstrip("/")
    m = json.load(open(MANIFEST))
    cks = m["checkpoints"]
    new13 = [c for c in cks if c["baseline"].startswith("new_")]
    jobs = []
    # Agreed order (2026-09-17): all frozen baselines first, then every GT run, Random last.
    # 1) frozen greedy + 32 samples on the 13 new/remeasured points, then the two 128-sample top-ups
    for c in interleave(new13):
        jobs.append(frozen_job(c, root))
    hp = m["high_precision_baseline"]
    for tag in hp["tags"]:
        c = next(x for x in cks if x["tag"] == tag)
        jobs.append(frozen_job(c, root, samples=len(hp["additional_sample_seeds"]),
                               seed_start=hp["additional_sample_seeds"][0], name="frozen_topup"))
    # 2) GT 500-step, seed-major over interleaved checkpoints
    for s in (31001, 31002, 31003, 31004):
        for c in interleave(cks):
            if s in c["new_gt_seeds"]:
                jobs.append(rl_job(c, "ground_truth", s, root))
    # 3) Random 500-step, seed-major over interleaved checkpoints
    for s in range(31001, 31009):
        for c in interleave(cks):
            if s in c["new_random_seeds"]:
                jobs.append(rl_job(c, "random", s, root))

    counts = {k: sum(1 for j in jobs if (j["kind"], j["key"]["reward_mode"])[1] == k) for k in ("random", "ground_truth", "frozen", "frozen_topup")}
    assert counts == {"random": m["workload"]["new_random_500"], "ground_truth": m["workload"]["new_gt_500"],
                      "frozen": m["workload"]["new_baselines"], "frozen_topup": m["workload"]["new_high_precision_topups"]}, counts

    out = pathlib.Path(root); (out / "logs").mkdir(parents=True, exist_ok=True)
    with open(out / "jobs.jsonl", "w") as f:
        for i, j in enumerate(jobs):
            f.write(json.dumps({"idx": i, **j}) + "\n")
    launch = dict(experiment_id=m["experiment_id"], manifest_sha256=sha(MANIFEST),
                  trainer=str(TRAINER.relative_to(REPO)), trainer_sha256=sha(TRAINER), frozen_eval_sha256=sha(FROZEN),
                  dataset=str(DATASET.relative_to(REPO)), dataset_sha256=sha(DATASET), model=MODEL, protocol_id=PROTOCOL_ID,
                  protocol=m["protocol"], modules=["baselines_13", "rl_500_new_204", "pair_topup_128"], optional_long_gt="not in this batch",
                  reuse="24h results reused elsewhere by the authors; this batch runs only the new work",
                  gpu_type="NVIDIA H200 (spec estimates assumed H100)", concurrency=args.concurrency, n_jobs=len(jobs), counts=counts,
                  argv_template=jobs[1]["cmd"], frozen_argv_template=jobs[0]["cmd"], planned_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(launch, open(out / "launch_manifest.json", "w"), indent=1)
    sbatch = f'''#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --array=0-{len(jobs) - 1}%{args.concurrency}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=4:00:00
# {m["experiment_id"]}: {len(jobs)} jobs ({counts}). One GPU each; finished jobs are skipped; re-submit failed ids with --array=<ids>.
set -uo pipefail
export HF_HOME=${{PROBE_HF_HOME:-${{HF_HOME:-$HOME/.cache/huggingface}}}} TOKENIZERS_PARALLELISM=false
cd {MS}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {root}/jobs.jsonl)
out=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["out"])' "$line")
marker=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["done_marker"])' "$line")
cmd=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["cmd"])' "$line")
if [[ -f $out ]] && grep -q "$marker" "$out"; then echo "skip: $out already complete"; exit 0; fi
mkdir -p "$(dirname "$out")"
attempt=$(( $(grep -c "\\"idx\\": $SLURM_ARRAY_TASK_ID," {root}/run_status.jsonl 2>/dev/null) + 1 ))
t0=$(date +%s); echo "[$(date -Is)] start idx=$SLURM_ARRAY_TASK_ID attempt=$attempt host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "$cmd"
eval "$cmd"; rc=$?
echo "{{\\"idx\\": $SLURM_ARRAY_TASK_ID, \\"attempt\\": $attempt, \\"job\\": \\"$SLURM_JOB_ID\\", \\"host\\": \\"$(hostname)\\", \\"start\\": $t0, \\"end\\": $(date +%s), \\"rc\\": $rc, \\"out\\": \\"$out\\"}}" >> {root}/run_status.jsonl
echo "[$(date -Is)] done rc=$rc"; exit $rc
'''
    (out / "run_array.sbatch").write_text(sbatch)
    print(f"{len(jobs)} jobs: {counts} -> {out}/jobs.jsonl, launch_manifest.json, run_array.sbatch")
    print("first 12 in queue:", [(j["tag"], j["kind"] if j["kind"] != "rl" else j["key"]["reward_mode"][:2] + str(j["key"]["seed"])[-1]) for j in jobs[:12]])
    print(f"submit (after review):  sbatch {out}/run_array.sbatch")


if __name__ == "__main__":
    main()
