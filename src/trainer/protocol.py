"""Pure helpers for the fixed-split checkpoint scan protocol."""

from __future__ import annotations

import hashlib
import json
import math
import random


def row_id(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_rows(rows: list[dict], train_frac: float, split_seed: int):
    indices = list(range(len(rows)))
    random.Random(split_seed).shuffle(indices)
    boundary = int(len(rows) * train_frac)
    train_indices, eval_indices = indices[:boundary], indices[boundary:]
    return (
        [rows[i] for i in train_indices],
        [rows[i] for i in eval_indices],
        [row_id(rows[i]) for i in train_indices],
        [row_id(rows[i]) for i in eval_indices],
    )


def pass_at_k(correct_samples: int, total_samples: int, k: int) -> float:
    """Unbiased pass@k estimator used by code-generation evaluations."""
    if not 0 <= correct_samples <= total_samples:
        raise ValueError("correct_samples must be between zero and total_samples")
    if not 1 <= k <= total_samples:
        raise ValueError("k must be between one and total_samples")
    if total_samples - correct_samples < k:
        return 1.0
    return 1.0 - math.comb(total_samples - correct_samples, k) / math.comb(total_samples, k)
