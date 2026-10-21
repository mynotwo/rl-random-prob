#!/usr/bin/env python3
"""SmolLM2 / OLMo 3 cross-model supplement: job list + sbatch array.

Phases: frozen (8 Raw baselines), smoke, rl (16 GT + 64 Random 500-step runs).
Writes files only; submit with  sbatch <root>/run_array.sbatch

Run with the venv python: it also tokenizes the 200 eval prompts with each model's own tokenizer
and asserts nothing is truncated and the eval ids are the OLMo 2 split-777 ids.
"""
import argparse, hashlib, json, pathlib, shlex, statistics, sys, time
import os

HERE = pathlib.Path(__file__).resolve().parent
MS = HERE.parent
REPO = MS.parent
SCAN = MS / "trainer"
FROZEN = SCAN / "frozen_eval.py"
DATASET = MS / "data" / "synthetic_gsm_rlvr_1000_seed13.jsonl"
PY = os.environ.get("PROBE_PYTHON", "python3")
OLMO2_REF = os.environ.get("PROBE_OUT", "runs") + "/checkpoint_curve_completion_20260915_20260917T0300Z/p1_839B/frozen.json"
SMOL, OLMO3 = "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints", "allenai/Olmo-3-1025-7B"
# branch + SHA verified against the HF refs API on 2026-09-20; weights are loaded by SHA, branch is the label.
CKPTS = [
    ("S1", SMOL, "step-125000", "47001ef69312fd851cd3c9a9212f324b5360b35d"),
    ("S2", SMOL, "step-2750000", "deff3e74e6cf4d947f4215cb0db6aadff34be270"),
    ("S3", SMOL, "step-4750000", "b5dd88a4b9b7cc3bf50ceed9115dc318cdee6939"),
    ("S4", SMOL, "step-5125000", "4916b26a1d198b4b6ad04b5c381a0f9fb1970aaa"),
    ("O1", OLMO3, "stage1-step8000", "ebfe7b62b31bfadff464a8926a48a256c9431db7"),
    ("O2", OLMO3, "stage1-step1413814", "373bad25002f1624757a73235c5ca844c6375c25"),
    ("O3", OLMO3, "stage2-step2000", "496e5c89b0229bbae667830a09b60708ca8eb297"),
    ("O4", OLMO3, "stage2-step47684", "c3c800dc900f3fecc112e8cd6b2a13edabc096e1"),
]
# Finished base models for the random-question control (Llama 3.1 / Qwen2.5 are in Shao et al. 2025; Qwen3 added by the authors).
# No intermediate checkpoints: branch is `main`, SHA as of 2026-09-20. Llama is gated (token in $HF_HOME/token).
SHAO = [
    ("L31_8B", "meta-llama/Llama-3.1-8B", "main", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
    ("Q25_7B", "Qwen/Qwen2.5-7B", "main", "d149729398750b98c0af14eb82c78cfe92750796"),
    ("Q3_8B", "Qwen/Qwen3-8B-Base", "main", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"),
]
TRAINER = SCAN / "rlvr_probe_v2_fixedsplit.py"      # the reference trainer, byte-identical to the OLMo 2 scan
KL = {"random": "0.01", "ground_truth": "0"}
GT_SEEDS, RANDOM_SEEDS, SMOKE_SEED = [31001, 31002], list(range(31001, 31009)), 39001
# counted on the meta device 2026-09-20 (r=8 on q/k/v/o/gate/up/down; every Linear except lm_head is covered)
LORA = {SMOL: dict(trainable=9.0e6, fraction=0.00528), OLMO3: dict(trainable=20.0e6, fraction=0.00274)}


def rl_job(c, reward, seed, root, steps=500, sub="runs"):
    cid, repo, branch, rev = c
    odir = f"{root}/{sub}/{cid}/{'gt' if reward == 'ground_truth' else reward}/seed{seed}"
    suffix = f"{cid}_{reward}_seed{seed}"
    cmd = [PY, str(TRAINER), "--dataset_path", str(DATASET), "--output_dir", odir, "--base_model", repo, "--revision", rev,
           "--train_mode", "original", "--reward_mode", reward, "--seed", str(seed), "--split_seed", "777",
           "--train_steps", str(steps), "--group_size", "4", "--max_new_tokens", "96", "--learning_rate", "3e-5",
           "--kl_coeff", KL[reward], "--eval_every", str(min(100, steps)), "--eval_limit", "200", "--eval_batch_size", "8",
           "--lora_r", "8", "--lora_alpha", "16", "--lora_dropout", "0", "--run_suffix", suffix, "--save_adapter"]
    return dict(kind=reward, id=cid, repo=repo, branch=branch, sha=rev, seed=seed, steps=steps, done_marker='"phase": "after"', cmd=shlex.join(cmd),
                out=f"{odir}/rlvr_{repo.split('/')[-1]}_{rev}_original_{reward}_seed{seed}_{suffix}.jsonl")


SETS = {"smollm2_olmo3_supplement_20260920": CKPTS, "shao_models_raw_20260920": SHAO}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def prompt_check(ckpts):
    """Same prompt path as frozen_eval.py; per-model token stats for the manifest."""
    sys.path[:0] = [str(MS), str(SCAN)]
    from transformers import AutoTokenizer
    from probe_checkpoints import fields_for_mode, load_jsonl
    from protocol import split_rows
    from rlvr_probe_v2_fixedsplit import ensure_padding_token
    _, eval_rows, _, eval_ids = split_rows(load_jsonl(str(DATASET)), 0.8, 777)
    if os.path.exists(OLMO2_REF):                     # the OLMo 2 scan's frozen file, when its results are present
        assert eval_ids == json.load(open(OLMO2_REF))["summary"]["eval_ids"], "eval ids differ from the OLMo 2 scan"
    prompts = ["{2}\n\nQuestion: {0}\nAnswer:".format(*fields_for_mode(r, "original")) for r in eval_rows]
    stats = {}
    for cid, repo, branch, rev in ckpts:
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, use_fast=True)
        ensure_padding_token(tok)
        n = [len(x) for x in tok(prompts)["input_ids"]]
        assert max(n) < 1024, (cid, max(n))          # frozen_eval truncates at 1024
        stats[cid] = dict(prompt_tokens_min=min(n), prompt_tokens_median=statistics.median(n), prompt_tokens_max=max(n),
                          pad_token=tok.pad_token, eos_token=tok.eos_token)
    return stats, hashlib.sha256(json.dumps(eval_ids).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--prompt_batch", type=int, default=8)   # spec: greedy eval batch size 8 (the RL trainer's eval batch)
    ap.add_argument("--phase", default="frozen", choices=["frozen", "smoke", "rl"],
                    help="smoke = 50-step GT run on S4 and O4 with seed 39001 (spec 4.1); rl = 16 GT then 64 Random 500-step runs (spec 4.2)")
    ap.add_argument("--set", default="smollm2_olmo3_supplement_20260920", choices=list(SETS))
    args = ap.parse_args()
    ckpts = SETS[args.set]
    root = args.out.rstrip("/")
    by_model = [[c for c in ckpts if c[1] == m] for m in (SMOL, OLMO3)]
    order = [c for pair in zip(*by_model) for c in pair] or ckpts   # S1,O1,S2,O2,... so both cards mix 1.7B and 7B; other sets keep list order
    jobs = []
    if args.phase == "smoke":
        jobs = [rl_job(c, "ground_truth", SMOKE_SEED, root, steps=50, sub="smoke") for c in order if c[0] in ("S4", "O4")]
    elif args.phase == "rl":    # agreed order: every GT run first, then Random; inside a block seed-major, checkpoints round-robin
        jobs = [rl_job(c, r, seed, root) for r, seeds in (("ground_truth", GT_SEEDS), ("random", RANDOM_SEEDS)) for seed in seeds for c in order]
    for cid, repo, branch, rev in (order if args.phase == "frozen" else []):
        out = f"{root}/frozen/{cid}/frozen.json"
        # no --sample_seed_start: 4 rounds x 8 return-sequences instead of 32 one-sample rounds (~8x fewer generate calls)
        cmd = [PY, str(FROZEN), "--dataset_path", str(DATASET), "--output", out, "--base_model", repo, "--revision", rev,
               "--tag", cid, "--split_seed", "777", "--samples", "32", "--prompt_batch", str(args.prompt_batch)]
        jobs.append(dict(kind="frozen", id=cid, repo=repo, branch=branch, sha=rev, out=out, done_marker='"summary"', cmd=shlex.join(cmd)))
    stats, ids_sha = prompt_check(ckpts)
    out = pathlib.Path(root); sfx = "" if args.phase == "frozen" else f"_{args.phase}"     # frozen keeps the original file names
    if (out / f"jobs{sfx}.jsonl").exists():
        raise SystemExit(f"{out}/jobs{sfx}.jsonl exists: refusing to overwrite a job list that may be running")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    with open(out / f"jobs{sfx}.jsonl", "w") as f:
        for i, j in enumerate(jobs):
            f.write(json.dumps({"idx": i, **j}) + "\n")
    json.dump(dict(experiment=args.set, phase=args.phase, checkpoints=[dict(zip(("id", "repo", "branch", "sha"), c)) for c in ckpts],
                   dataset=str(DATASET.relative_to(REPO)), dataset_sha256=sha(DATASET), split_seed=777, eval_ids_sha256=ids_sha,
                   eval_ids_match_olmo2_scan=True, frozen_eval_sha256=sha(FROZEN), prompt_template="{instruction}\\n\\nQuestion: {question}\\nAnswer:",
                   eval=dict(greedy_batch=args.prompt_batch, max_new_tokens=96, samples=32, temperature=1.0, top_p=0.95,
                             note="sampling layout is 4 rounds x 8 return-sequences, seed 777; OLMo 2 scan used 32 per-seed rounds (41001+). Greedy is unaffected."),
                   rl=None if args.phase == "frozen" else dict(trainer=str(TRAINER.relative_to(REPO)), trainer_sha256=sha(TRAINER), gt_seeds=GT_SEEDS, random_seeds=RANDOM_SEEDS,
                       smoke_seed=SMOKE_SEED, lora_targets="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", lora_trainable=LORA,
                       protocol="LoRA r8/a16/dropout 0, lr 3e-5, G 4, 500 steps, 96 new tokens, kl 0 (GT) / 0.01 (Random), Random = Bernoulli(0.5), eval 200 items every 100 steps, greedy batch 8, step-500 adapter saved",
                       not_logged_vs_spec_6="gradient norm, peak GPU memory, per-item eval text (only 3 examples per eval); step-500 per-item outputs are recoverable from the saved adapters"),
                   tokenizer_prompt_stats=stats, gpu_type="NVIDIA H200 (spec text says H100)", planned=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
              open(out / f"manifest{sfx}.json", "w"), indent=1)
    (out / f"run_array{sfx}.sbatch").write_text(f'''#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --array=0-{len(jobs) - 1}%{args.concurrency}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time={"1:30:00" if args.phase != "rl" else "3:00:00"}
# {args.set} / {args.phase}: {len(jobs)} jobs, one GPU each. Finished jobs are skipped.
set -uo pipefail
export HF_HOME=${{PROBE_HF_HOME:-${{HF_HOME:-$HOME/.cache/huggingface}}}} TOKENIZERS_PARALLELISM=false
cd {MS}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {root}/jobs{sfx}.jsonl)
out=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["out"])' "$line")
marker=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["done_marker"])' "$line")
cmd=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["cmd"])' "$line")
if [[ -f $out ]] && grep -q "$marker" "$out"; then echo "skip: $out already complete"; exit 0; fi
mkdir -p "$(dirname "$out")"
sleep $((RANDOM % {1 if args.phase == "frozen" else 45}))   # stagger: many tasks cold-starting CUDA on one node at once hit context-creation OOM (2026-09-17)
for attempt in 1 2; do
  t0=$(date +%s); echo "[$(date -Is)] start idx=$SLURM_ARRAY_TASK_ID attempt=$attempt host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  {PY} -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
  echo "$cmd"
  eval "$cmd" 2> >(tee /tmp/exp_err_$SLURM_JOB_ID >&2); rc=$?
  echo "{{\\"idx\\": $SLURM_ARRAY_TASK_ID, \\"attempt\\": $attempt, \\"job\\": \\"$SLURM_JOB_ID\\", \\"host\\": \\"$(hostname)\\", \\"start\\": $t0, \\"end\\": $(date +%s), \\"rc\\": $rc, \\"out\\": \\"$out\\"}}" >> {root}/run_status{sfx}.jsonl
  # retry once only for the infrastructure failure seen on 2026-09-17: CUDA OOM while creating the context
  if [[ $rc -ne 0 && $attempt -eq 1 ]] && grep -q "out of memory" /tmp/exp_err_$SLURM_JOB_ID && ! grep -qs '"phase": "train"' "$out"; then echo "load-time CUDA OOM, retrying once in 60 s"; rm -f "$out"; sleep 60; continue; fi
  break
done
rm -f /tmp/exp_err_$SLURM_JOB_ID
echo "[$(date -Is)] done rc=$rc"; exit $rc
''')
    print(f"{len(jobs)} jobs -> {out}/jobs{sfx}.jsonl, manifest{sfx}.json, run_array{sfx}.sbatch | queue head: {[(j['id'], j['kind'][:2], j.get('seed', 0) % 100) for j in jobs[:16]]}")
    for cid in stats:
        print(f"  {cid} prompt tokens: {stats[cid]}")
    print(f"submit (after review):  sbatch {out}/run_array{sfx}.sbatch")


if __name__ == "__main__":
    main()
