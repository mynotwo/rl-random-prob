#!/usr/bin/env python3
"""Greedy plus 32-sample frozen-model evaluation for the 10h scan."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_checkpoints import extract_final_number, extract_strict_hash_number, fields_for_mode, load_jsonl
from protocol import pass_at_k, split_rows
from rlvr_probe_v2_fixedsplit import ensure_padding_token, filter_generate_inputs


def is_correct(value, gold):
    return value is not None and gold is not None and math.isclose(value, gold, rel_tol=0, abs_tol=1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split_seed", type=int, default=777)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--sample_seed_start", type=int, default=None,
                    help="If set, generate one sample per prompt with each consecutive seed")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--prompt_batch", type=int, default=4)
    ap.add_argument("--sample_batch", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    torch.cuda.manual_seed_all(args.split_seed)
    rows = load_jsonl(args.dataset_path)
    _, eval_rows, _, eval_ids = split_rows(rows, 0.8, args.split_seed)
    prompts, golds = [], []
    for row in eval_rows:
        question, _, instruction = fields_for_mode(row, "original")
        prompts.append(f"{instruction}\n\nQuestion: {question}\nAnswer:")
        golds.append(extract_final_number(row["orig_answer"]))

    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision, use_fast=True)
    ensure_padding_token(tok)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.revision, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    records = [{"eval_id": rid, "gold": gold, "greedy": None, "samples": []}
               for rid, gold in zip(eval_ids, golds)]
    started = time.time()

    def generate(batch_prompts, do_sample, num_return_sequences):
        enc = tok(batch_prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        enc = filter_generate_inputs(model, enc)
        kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=do_sample,
                      num_return_sequences=num_return_sequences,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        if do_sample:
            kwargs.update(temperature=args.temperature, top_p=args.top_p)
        out = model.generate(**enc, **kwargs)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    with torch.no_grad():
        for start in range(0, len(prompts), args.prompt_batch):
            batch = prompts[start:start + args.prompt_batch]
            texts = generate(batch, False, 1)
            for offset, text in enumerate(texts):
                records[start + offset]["greedy"] = text
        per_round = 1 if args.sample_seed_start is not None else args.sample_batch
        rounds = args.samples if args.sample_seed_start is not None else args.samples // per_round
        if rounds * per_round != args.samples:
            raise ValueError("samples must be divisible by samples generated per round")
        for round_idx in range(rounds):
            if args.sample_seed_start is not None:
                sample_seed = args.sample_seed_start + round_idx
                random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
            for start in range(0, len(prompts), args.prompt_batch):
                batch = prompts[start:start + args.prompt_batch]
                texts = generate(batch, True, per_round)
                for offset in range(len(batch)):
                    lo, hi = offset * per_round, (offset + 1) * per_round
                    records[start + offset]["samples"].extend(texts[lo:hi])
            print(f"[sample round {round_idx + 1}/{rounds}] elapsed={time.time()-started:.0f}s", flush=True)

    for rec in records:
        gold = rec["gold"]
        greedy = rec["greedy"]
        rec["greedy_lenient_correct"] = is_correct(extract_final_number(greedy), gold)
        rec["greedy_strict_correct"] = is_correct(extract_strict_hash_number(greedy), gold)
        rec["sample_lenient_correct"] = [is_correct(extract_final_number(x), gold) for x in rec["samples"]]
        rec["sample_strict_correct"] = [is_correct(extract_strict_hash_number(x), gold) for x in rec["samples"]]

    summary = {
        "tag": args.tag, "base_model": args.base_model, "revision": args.revision,
        "split_seed": args.split_seed, "eval_ids": eval_ids, "n_prompts": len(records),
        "samples_per_prompt": args.samples, "temperature": args.temperature, "top_p": args.top_p,
        "sample_seeds": (list(range(args.sample_seed_start, args.sample_seed_start + args.samples))
                         if args.sample_seed_start is not None else None),
        "max_new_tokens": args.max_new_tokens, "elapsed_seconds": time.time() - started,
    }
    for flavor in ("lenient", "strict"):
        counts = [sum(rec[f"sample_{flavor}_correct"]) for rec in records]
        summary[f"greedy_{flavor}_accuracy"] = sum(rec[f"greedy_{flavor}_correct"] for rec in records) / len(records)
        summary[f"sample_{flavor}_accuracy"] = sum(counts) / (len(records) * args.samples)
        for k in (1, 4, 8, 16, 32):
            summary[f"pass_at_{k}_{flavor}"] = sum(pass_at_k(c, args.samples, k) for c in counts) / len(counts)
    all_samples = [x for rec in records for x in rec["samples"]]
    summary["sample_format_rate"] = sum("####" in x or "\\boxed" in x for x in all_samples) / len(all_samples)
    summary["sample_mean_whitespace_tokens"] = sum(len(x.split()) for x in all_samples) / len(all_samples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, ensure_ascii=False)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
