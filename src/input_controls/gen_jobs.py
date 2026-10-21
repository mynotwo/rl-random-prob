#!/usr/bin/env python3
"""Random question + random reward: arms C / A / D x 16 seeds at one checkpoint, evaluated on the GSM
eval set of the checkpoint-curve runs. Reference trainer and argv unchanged (24h protocol, random reward,
kl 0.01); only --dataset_path differs. Arm B (GSM questions) is NOT queued: it exists in the 24h results.

    python input_controls/gen_jobs.py --tag p2_50B --out runs/input_controls --concurrency 32
"""
from __future__ import annotations
import os

import argparse, hashlib, json, pathlib, shlex, time

HERE = pathlib.Path(__file__).resolve().parent
MS = HERE.parent
REPO = MS.parent
TRAINER = MS / "trainer" / "rlvr_probe_v2_fixedsplit.py"
GSM_MANIFEST = REPO / "configs" / "olmo2_checkpoints.json"
MODEL = "allenai/OLMo-2-1124-7B"
PY = os.environ.get("PROBE_PYTHON", "python3")
# B = the unmodified GSM training questions (the GSM scan's own Random protocol); seeds 31001-31008 of B exist in the 24h results
ARMS = {"B": "synthetic_gsm_rlvr_1000_seed13", "C": "randq_C_shuffled_gsm", "A": "randq_A_mmlu", "D": "randq_D_randtok", "E": "randq_E_rawtok"}
# arm E needs prompts with no template at all -> the raw-prompt copy of the trainer (one added branch in build_prompt;
# byte-identical records to the reference script on GSM data, checked on CPU 2026-09-20)
RAW_TRAINER = MS / "trainer" / "rlvr_probe_v2_fixedsplit_rawprompt.py"
TRAINER_OF = {"E": RAW_TRAINER}
SEEDS = list(range(31001, 31017))     # first batch; later batches pass --seeds
# The Qwen models of Shao et al. 2025 ("Spurious Rewards"). No intermediate checkpoints exist; SHA = `main` on 2026-09-20.
# The D/E datasets are reused unchanged: their prompts tokenize to the same lengths under the Qwen tokenizer (median 55 vs GSM 55).
EXTRA = {"qwen25_math_7b": ("Qwen/Qwen2.5-Math-7B", "b101308fe89651ea5ce025f25317fea6fc07e96e"),
         "qwen25_math_1p5b": ("Qwen/Qwen2.5-Math-1.5B", "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"),
         "qwen25_7b": ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"),
         "qwen25_1p5b": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
         # also in Shao et al. (gated: needs the token in $HF_HOME/token); Qwen3 is NOT in their paper, added by the authors
         "llama31_8b": ("meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
         "qwen3_8b_base": ("Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4")}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["p2_50B"], help="OLMo 2 checkpoint tag(s) from the GSM manifest, or keys of EXTRA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--arms", nargs="+", default=["C", "A", "D"], choices=list(ARMS))
    ap.add_argument("--seeds", nargs=2, type=int, metavar=("FIRST", "LAST"), help="inclusive seed range; default 31001 31016")
    args = ap.parse_args()
    seeds = list(range(args.seeds[0], args.seeds[1] + 1)) if args.seeds else SEEDS
    root = args.out.rstrip("/")
    gsm_cks = {c["tag"]: c for c in json.load(open(GSM_MANIFEST))["checkpoints"]}
    # model, revision to load, checkpoint record; OLMo 2 loads by branch (as every earlier run), EXTRA models by SHA
    cks = {t: (EXTRA[t][0], EXTRA[t][1], dict(tag=t, repo=EXTRA[t][0], branch="main", sha=EXTRA[t][1])) if t in EXTRA
           else (MODEL, gsm_cks[t]["branch"], gsm_cks[t]) for t in args.tag}
    jobs = []
    for seed, tag, arm in ((s, t, a) for s in seeds for t in args.tag for a in args.arms):   # seed-major, then models, then arms: every wave mixes them
        if True:
            model, rev, ck = cks[tag]; adir = f"{root}/{arm}" if len(args.tag) == 1 else f"{root}/{tag}/{arm}"
            stem = ARMS[arm]; trainer = TRAINER_OF.get(arm, TRAINER)
            suffix = f"{tag}_randq{arm}_random_seed{seed}"
            cmd = [PY, str(trainer), "--dataset_path", str(MS / "data" / f"{stem}.jsonl"), "--output_dir", adir,
                   "--base_model", model, "--revision", rev, "--train_mode", "original",
                   "--reward_mode", "random", "--seed", str(seed), "--split_seed", "777",
                   "--train_steps", "500", "--group_size", "4", "--max_new_tokens", "96",
                   "--learning_rate", "3e-5", "--kl_coeff", "0.01", "--eval_every", "100",
                   "--eval_limit", "200", "--eval_batch_size", "8", "--lora_r", "8",
                   "--lora_alpha", "16", "--lora_dropout", "0", "--run_suffix", suffix, "--save_adapter"]
            jobs.append(dict(arm=arm, tag=tag, seed=seed, key=dict(checkpoint_sha=ck["sha"], arm=arm, reward_mode="random", seed=seed, steps=500),
                             out=f"{adir}/rlvr_{model.split('/')[-1]}_{rev}_original_random_seed{seed}_{suffix}.jsonl", cmd=shlex.join(cmd)))
    out = pathlib.Path(root)
    if (out / "jobs.jsonl").exists():
        raise SystemExit(f"{out}/jobs.jsonl exists: refusing to overwrite a job list that may be running")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    with open(out / "jobs.jsonl", "w") as f:
        for i, j in enumerate(jobs):
            f.write(json.dumps({"idx": i, **j}) + "\n")
    json.dump(dict(experiment="random_question_random_reward", checkpoint=cks[args.tag[0]][2] if len(args.tag) == 1 else None, checkpoints=[c[2] for c in cks.values()], arms={a: dict(dataset=f"data/{ARMS[a]}.jsonl", sha256=sha(MS / "data" / f"{ARMS[a]}.jsonl"), trainer=str(TRAINER_OF.get(a, TRAINER).relative_to(REPO)), trainer_sha256=sha(TRAINER_OF.get(a, TRAINER))) for a in args.arms},
                   arm_B="GSM questions + random reward: seeds 31001-31008 are in the 24h results and are never re-run here",
                   datasets_report={**json.load(open(HERE / "datasets_report.json")), "E": json.load(open(HERE / "dataset_E_report.json")) if (HERE / "dataset_E_report.json").exists() else None}, seeds=seeds, trainer=str(TRAINER.relative_to(REPO)), trainer_sha256=sha(TRAINER),
                   protocol="24h: lr 3e-5, random kl 0.01, G 4, 500 steps, 96 tokens, split 777, eval 200 GSM items every 100 steps",
                   decision_rule="count seeds with terminal delta > 0.10 per arm; Fisher exact vs arm B; report terminal max, mid-run peak, rollout length and format rate; B at 0 jackpots voids the test",
                   n_jobs=len(jobs), concurrency=args.concurrency, planned_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(out / "manifest.json", "w"), indent=1, ensure_ascii=False)
    sb = f'''#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --array=0-{len(jobs) - 1}%{args.concurrency}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=3:00:00
# random question + random reward @ {"/".join(args.tag)}: {len(jobs)} runs (arms {"/".join(args.arms)} x {len(seeds)} seeds). Finished runs are skipped.
set -uo pipefail
export HF_HOME=${{PROBE_HF_HOME:-${{HF_HOME:-$HOME/.cache/huggingface}}}} TOKENIZERS_PARALLELISM=false
cd {MS}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {root}/jobs.jsonl)
out=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["out"])' "$line")
cmd=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["cmd"])' "$line")
if [[ -f $out ]] && grep -q '"phase": "after"' "$out"; then echo "skip: $out already complete"; exit 0; fi
mkdir -p "$(dirname "$out")"
sleep $((RANDOM % 45))   # stagger: many tasks cold-starting CUDA on one idle node at once hit context-creation OOM (2026-09-17)
for attempt in 1 2; do
  t0=$(date +%s); echo "[$(date -Is)] start idx=$SLURM_ARRAY_TASK_ID attempt=$attempt host=$(hostname)"
  echo "$cmd"
  eval "$cmd" 2> >(tee /tmp/exp_err_$SLURM_JOB_ID >&2); rc=$?
  echo "{{\\"idx\\": $SLURM_ARRAY_TASK_ID, \\"attempt\\": $attempt, \\"job\\": \\"$SLURM_JOB_ID\\", \\"host\\": \\"$(hostname)\\", \\"start\\": $t0, \\"end\\": $(date +%s), \\"rc\\": $rc, \\"out\\": \\"$out\\"}}" >> {root}/run_status.jsonl
  # retry once only for the infrastructure failure: CUDA OOM before any training record exists
  if [[ $rc -ne 0 && $attempt -eq 1 ]] && grep -q "out of memory" /tmp/exp_err_$SLURM_JOB_ID && ! grep -qs '"phase": "train"' "$out"; then
    echo "load-time CUDA OOM, retrying once in 60 s"; rm -f "$out"; sleep 60; continue
  fi
  break
done
rm -f /tmp/exp_err_$SLURM_JOB_ID
echo "[$(date -Is)] done rc=$rc"; exit $rc
'''
    (out / "run_array.sbatch").write_text(sb)
    print(f"{len(jobs)} jobs -> {out}/jobs.jsonl ; first 6: {[(j['arm'], j['seed']) for j in jobs[:6]]}")
    print(f"submit (after review):  sbatch {out}/run_array.sbatch")


if __name__ == "__main__":
    main()
