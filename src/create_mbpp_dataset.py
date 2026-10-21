#!/usr/bin/env python3
"""Convert MBPP to RLVR-compatible JSONL format."""
import json
import os

from datasets import load_dataset

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mbpp_rlvr_964.jsonl")

records = []
for split in ["train", "test", "validation"]:
    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    for ex in ds:
        rec = {
            "orig_question": ex["text"],
            "orig_answer": ex["code"],
            "test_list": ex["test_list"],
            "test_setup_code": ex.get("test_setup_code", ""),
            "challenge_test_list": ex.get("challenge_test_list", []),
            "task_id": ex["task_id"],
        }
        records.append(rec)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"Wrote {len(records)} examples to {out_path}")
