#!/usr/bin/env python3
"""Task-level worker for the 37-checkpoint OLMo-2 scan."""

from __future__ import annotations

# --- portability shim (added when consolidating 138/141) ---
# Override with the PROBE_REPO / PROBE_OUT environment variables.
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("PROBE_REPO", _Path(__file__).resolve().parents[2]))
PROBE_OUT = _Path(_os.environ.get("PROBE_OUT", REPO_ROOT / "experiments"))
# ----------------------------------------------------------
import argparse
import json
import pathlib
import subprocess
import sys
import time

MODEL = "allenai/OLMo-2-1124-7B"
DATASET = str(REPO_ROOT) + "/src/data/synthetic_gsm_rlvr_1000_seed13.jsonl"


def build_contract():
    contract = [("frozen", None, None),
                ("random_seed31001", ("random", 31001), "0.01"),
                ("ground_truth_seed31001", ("ground_truth", 31001), "0"),
                ("random_seed31002", ("random", 31002), "0.01"),
                ("ground_truth_seed31002", ("ground_truth", 31002), "0")]
    contract += [(f"random_seed{seed}", ("random", seed), "0.01")
                 for seed in range(31003, 31009)]
    return contract


def run_component(command, log_path, deadline):
    remaining = deadline - time.time()
    if remaining <= 2700:
        return 125
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"COMMAND {json.dumps(command)}\n")
        log.flush()
        try:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=remaining - 60).returncode
        except subprocess.TimeoutExpired:
            log.write("HARD DEADLINE REACHED\n")
            return 124


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--tags", required=True)
    args = ap.parse_args()
    scan = pathlib.Path(args.scan_dir)
    shared = scan.parent / "trainer"
    out = pathlib.Path(args.output_root)
    manifest = json.loads((scan / "manifest.json").read_text())
    by_tag = {c["tag"]: c for c in manifest["checkpoints"]}
    checkpoints = [by_tag[tag] for tag in args.tags.split(",") if tag]
    claims, done, logs, results = [out / name for name in ("claims", "done", "logs", "results")]
    for path in (claims, done, logs, results):
        path.mkdir(parents=True, exist_ok=True)

    contract = build_contract()
    tasks = [(checkpoint, spec) for spec in contract for checkpoint in checkpoints]

    for checkpoint, (name, train_spec, kl) in tasks:
        if time.time() >= args.deadline - 2700:
            break
        tag, revision = checkpoint["tag"], checkpoint["commit"]
        checkpoint_out = results / tag
        checkpoint_out.mkdir(exist_ok=True)
        if train_spec is not None and not (checkpoint_out / "frozen.done").exists():
            continue
        marker = checkpoint_out / f"{name}.done"
        if marker.exists():
            continue
        claim = claims / f"{tag}__{name}"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        (claim / "authors.json").write_text(json.dumps({"worker": args.worker, "time": time.time()}))
        if train_spec is None:
            command = [sys.executable, str(shared / "frozen_eval.py"),
                "--dataset_path", DATASET, "--output", str(checkpoint_out / "frozen.json"),
                "--base_model", MODEL, "--revision", revision, "--tag", tag,
                "--split_seed", "777", "--samples", "32", "--sample_seed_start", "41001"]
        else:
            reward, seed = train_spec
            suffix = f"{tag}_{reward}_seed{seed}"
            command = [sys.executable, str(shared / "rlvr_probe_v2_fixedsplit.py"),
                "--dataset_path", DATASET, "--output_dir", str(checkpoint_out),
                "--base_model", MODEL, "--revision", revision, "--train_mode", "original",
                "--reward_mode", reward, "--seed", str(seed), "--split_seed", "777",
                "--train_steps", "500", "--group_size", "4", "--max_new_tokens", "96",
                "--learning_rate", "3e-5", "--kl_coeff", kl, "--eval_every", "100",
                "--eval_limit", "200", "--eval_batch_size", "8", "--lora_r", "8",
                "--lora_alpha", "16", "--lora_dropout", "0", "--run_suffix", suffix,
                "--save_adapter"]
        rc = run_component(command, logs / f"{tag}_{name}.log", args.deadline)
        if rc:
            with open(out / "failed.tsv", "a", encoding="utf-8") as handle:
                handle.write(f"{tag}\t{name}\t{rc}\t{time.time()}\n")
            continue
        marker.write_text(f"completed_at={time.time()}\n")
        if all((checkpoint_out / f"{component_name}.done").exists()
               for component_name, _, _ in contract):
            (done / tag).write_text(f"completed_at={time.time()}\n")


if __name__ == "__main__":
    main()
