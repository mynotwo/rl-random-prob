#!/usr/bin/env python3
"""Shared-rollout 2x2 experiment: manifest, code snapshot, split files, job lists and sbatch arrays. Writes files only.

    gen_jobs.py --root ROOT --phase smoke      # replicate 59001, 20 groups / 20 steps per cell, 2 tasks
    gen_jobs.py --root ROOT --phase main       # replicates 51001-51016, 500 steps, 32 tasks (one per replicate x source)

One task = one source stream and its two learners (prepare S, train S->L1, train S->L2) on one GPU, so the two cells of a
row always share the cache that was just written. Tasks are ordered replicate-major: complete 2x2 blocks finish first.
Submit (after review):  sbatch ROOT/run_array_<phase>.sbatch
"""
import argparse, json, pathlib, shlex, shutil, subprocess, time
import os

import sr_common as C

HERE = pathlib.Path(__file__).resolve().parent
SNAPSHOT = [HERE / f for f in ("sr_common.py", "prepare_shared_rollouts.py", "train_shared_rollouts.py", "test_shared_rollouts.py", "gen_jobs.py")] + \
           [C.MS / "trainer" / "rlvr_probe_v2_fixedsplit.py", C.MS / "trainer" / "protocol.py", C.MS / "probe_checkpoints.py"]
STATS_PLAN = dict(
    unit="one replicate block (rollout streams + reward table + adapter start); never an eval item or an eval step",
    primary_family_holm=["sign test on d = gain(S1->L2) - gain(S1->L1)", "sign test on d = gain(S2->L2) - gain(S2->L1)",
                         "exact McNemar on improved (gain > 0.10), S1 row", "exact McNemar on improved, S2 row",
                         "ADDED 2026-09-21 before any formal run: sign test on |gain(S1->L2)| - |gain(S1->L1)|",
                         "ADDED 2026-09-21 before any formal run: sign test on |gain(S2->L2)| - |gain(S2->L1)|"],
    why_added="on-policy Random at P2 moves in both directions (collapses as well as jackpots); a directional test alone can miss a learner that responds much more",
    also="mean/median gain difference with block bootstrap (10,000 resamples, seed 20260921); Clopper-Pearson 95% per cell; source effect and interaction are exploratory",
    missing="never filled with zero, never replaced by a new seed; binary rates get an all-fail / all-success sensitivity range")
DEVIATIONS = ["mid-run resume (spec 8.8) replaced by deterministic rerun from step 0 under the same job key with an attempt counter: the training path draws no random numbers",
              "source generation is batched (10 prompts x 4 samples per generate call) instead of one prompt per call; fixed here, part of the stream's RNG layout",
              "no git commit is made (authors' rule): code_snapshot/ + sha256 + git HEAD/status stand in for it",
              "hardware is NVIDIA H200, not the H100 the spec's estimates assume",
              "source token log-probs are not cached (optional in the spec); Chen's section-5 quantity is left to the audit from rewards x correctness"]


def task(root, b, s, steps, groups, eval_every):
    base = [C.PY]; r = ["--root", root, "--replicate", str(b)]
    cmds = [base + [str(HERE / "prepare_shared_rollouts.py"), "--source", s, "--groups", str(groups)] + r] + \
           [base + [str(HERE / "train_shared_rollouts.py"), "--source", s, "--learner", l, "--steps", str(steps), "--eval_every", str(eval_every)] + r for l in C.LEARNERS]
    return dict(replicate=b, source=s, cmds=[shlex.join(c) for c in cmds], done=[f"{root}/runs/{s}_{l}/replicate{b}/summary.json" for l in C.LEARNERS])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--phase", required=True, choices=["smoke", "main"])
    ap.add_argument("--concurrency", type=int, default=16); a = ap.parse_args()
    top = pathlib.Path(a.root.rstrip("/")); root = str(top / "smoke") if a.phase == "smoke" else str(top)
    if (top / f"jobs_{a.phase}.jsonl").exists():
        raise SystemExit(f"{top}/jobs_{a.phase}.jsonl exists: refusing to overwrite a job list that may be running")
    (top / "logs").mkdir(parents=True, exist_ok=True)
    reps, steps = ([C.SMOKE_REPLICATE], 20) if a.phase == "smoke" else (C.REPLICATES, C.STEPS)
    jobs = [task(root, b, s, steps, steps, min(100, steps)) for b in reps for s in C.SOURCES]

    snap = top / "code_snapshot"; snap.mkdir(exist_ok=True)
    for f in SNAPSHOT:
        shutil.copy(f, snap)
    train_rows, eval_rows, train_ids, eval_ids = C.load_split()
    (top / "splits").mkdir(exist_ok=True)
    (top / "splits" / "train_ids_in_order.json").write_text(json.dumps(train_ids)); (top / "splits" / "eval_ids_in_order.json").write_text(json.dumps(eval_ids))
    hub = pathlib.Path(os.environ.get("PROBE_HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))) / "hub/models--allenai--OLMo-2-1124-7B/snapshots"
    files = {k: {f: (C.sha_file(hub / sha / f) if (hub / sha / f).exists() else None) for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json")} for k, (_, sha) in C.CKPT.items()}
    assert files["P1"] == files["P2"], "the two checkpoints do not share tokenizer/config files"
    git = lambda *c: subprocess.run(["git", "-C", str(C.MS), *c], capture_output=True, text=True).stdout.strip()
    manifest = dict(experiment="shared_rollout_cross_20260921", protocol_id=C.PROTOCOL_ID, phase=a.phase, model=C.MODEL,
                    checkpoints={k: dict(branch=br, sha=sha, role=[n for n, v in {**C.SOURCES, **C.LEARNERS}.items() if v == k]) for k, (br, sha) in C.CKPT.items()},
                    tokenizer_and_config_sha256=files["P1"], same_files_for_both_checkpoints=True,
                    dataset=dict(file=str(C.DATASET.relative_to(C.MS.parent)), sha256=C.DATASET_SHA, split_seed=777, n_train=len(train_ids), n_eval=len(eval_ids),
                                 train_ids_sha256=C.sha_obj(train_ids), eval_ids_sha256=C.sha_obj(eval_ids), schedule="step t reads train_rows[(t-1) % 800]: 500 steps visit the first 500 train rows once"),
                    replicates=reps, seeds=dict(rollout_S1="100000+b", rollout_S2="200000+b", reward="300000+b (CPU generator, 500x4 table)", lora_init="400000+b (CPU generator)", eval="greedy, no RNG"),
                    generation=dict(groups=steps, G=C.G, temperature=1.0, top_p=0.95, max_new_tokens=C.MAX_NEW, prompt_batch=10, retries="none", filtering="none"),
                    learner=dict(loss="-(sum_ij m_ij A_i log pi(y_ij)) / sum_ij m_ij, FP32 log-probs; no ratio, importance weight, PPO clipping, KL, entropy or SFT term",
                                 mask="cached response tokens that are not pad; OLMo 2 pad (100277) != eos (100257), so a generated EOS is inside the mask",
                                 steps=steps, lr=3e-5, schedule="constant, no warmup", optimizer="AdamW betas=(0.9,0.999) eps=1e-8 weight_decay=0", grad_norm_cap=1.0,
                                 lora="r8 alpha16 dropout0 bias none on q,k,v,o,gate,up,down; base frozen", precision="BF16 model, FP32 LoRA weights and log-probs",
                                 gradient_checkpointing=False, zero_advantage_steps="executed (backward + optimizer.step), never skipped",
                                 reference_differences="the September Random protocol had kl 0.01 and AdamW's default weight_decay 0.01; both are 0 here by design"),
                    evaluation=dict(steps=[0] + list(range(min(100, steps), steps + 1, min(100, steps))), decoding="greedy", max_new_tokens=C.MAX_NEW, batch=C.EVAL_BATCH, items=len(eval_ids),
                                    saved="every item's text and token ids", diagnostic="cutq_*: lenient score on the text before the next 'Question:' (not a blank-line cut); never the main metric"),
                    job_keys=[dict(source_sha=C.CKPT[C.SOURCES[s]][1], learner_sha=C.CKPT[C.LEARNERS[l]][1], replicate_id=b, protocol_id=C.PROTOCOL_ID, steps=steps) for b in reps for s in C.SOURCES for l in C.LEARNERS],
                    statistics_plan=STATS_PLAN, deviations_from_spec=DEVIATIONS, code_snapshot_sha256={f.name: C.sha_file(f) for f in SNAPSHOT},
                    git=dict(head=git("rev-parse", "HEAD"), dirty_files=len(git("status", "--porcelain").splitlines())), gpu_type="NVIDIA H200",
                    planned_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (top / f"manifest_{a.phase}.json").write_text(json.dumps(manifest, indent=1))
    with open(top / f"jobs_{a.phase}.jsonl", "w") as f:
        for i, j in enumerate(jobs):
            f.write(json.dumps({"idx": i, **j}) + "\n")
    (top / f"run_array_{a.phase}.sbatch").write_text(f'''#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --array=0-{len(jobs) - 1}%{a.concurrency}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time={"1:00:00" if a.phase == "smoke" else "4:00:00"}
# shared_rollout_cross_20260921 / {a.phase}: {len(jobs)} tasks, each = one source stream + its two learners. Finished pieces are skipped.
set -uo pipefail
export HF_HOME=${{PROBE_HF_HOME:-${{HF_HOME:-$HOME/.cache/huggingface}}}} TOKENIZERS_PARALLELISM=false
cd {HERE}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {top}/jobs_{a.phase}.jsonl)
n=$(python3 -c 'import json,sys; print(len(json.loads(sys.argv[1])["cmds"]))' "$line")
sleep $((RANDOM % 45))   # stagger: many tasks cold-starting CUDA on one node at once hit context-creation OOM (2026-09-17)
echo "[$(date -Is)] start idx=$SLURM_ARRAY_TASK_ID host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
rc=0
for k in $(seq 0 $((n - 1))); do
  cmd=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["cmds"][int(sys.argv[2])])' "$line" $k)
  for attempt in 1 2; do
    t0=$(date +%s); echo "$cmd"
    eval "$cmd" 2> >(tee /tmp/exp_err_$SLURM_JOB_ID >&2); rc=$?
    echo "{{\\"idx\\": $SLURM_ARRAY_TASK_ID, \\"cmd\\": $k, \\"attempt\\": $attempt, \\"job\\": \\"$SLURM_JOB_ID\\", \\"host\\": \\"$(hostname)\\", \\"start\\": $t0, \\"end\\": $(date +%s), \\"rc\\": $rc}}" >> {top}/run_status_{a.phase}.jsonl
    # retry once only for the infrastructure failure: CUDA OOM while the model is being loaded
    if [[ $rc -ne 0 && $attempt -eq 1 ]] && grep -q "out of memory" /tmp/exp_err_$SLURM_JOB_ID; then echo "CUDA OOM, retrying once in 60 s"; sleep 60; continue; fi
    break
  done
  [[ $rc -ne 0 ]] && break      # a learner must not start if its cache failed
done
rm -f /tmp/exp_err_$SLURM_JOB_ID
echo "[$(date -Is)] done rc=$rc"; exit $rc
''')
    print(f"{len(jobs)} tasks -> {top}/jobs_{a.phase}.jsonl, manifest_{a.phase}.json, run_array_{a.phase}.sbatch | first: {[(j['replicate'], j['source']) for j in jobs[:4]]}")
    print(f"submit:  sbatch {top}/run_array_{a.phase}.sbatch")


if __name__ == "__main__":
    main()
