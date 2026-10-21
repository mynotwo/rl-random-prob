#!/usr/bin/env python3
"""Arm E: the training prompt is ONLY uniformly random vocabulary tokens -- no instruction, no "Question:",
no "Answer:". Each of the 800 train rows gets `raw_prompt` whose token count equals that of the FULL templated
GSM prompt it replaces (OLMo-2 tokenizer). The 200 eval rows are copied verbatim (no raw_prompt), so
rlvr_probe_v2_fixedsplit_rawprompt.py evaluates them through the original template: same eval_ids as every GSM run."""
from __future__ import annotations

import hashlib, json, pathlib, random, statistics, sys

HERE = pathlib.Path(__file__).resolve().parent
MS = HERE.parent
sys.path.insert(0, str(MS / "trainer")); sys.path.insert(0, str(MS))
from protocol import split_rows  # noqa: E402
from probe_checkpoints import fields_for_mode  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

GSM = MS / "data" / "synthetic_gsm_rlvr_1000_seed13.jsonl"
REV = "stage2-ingredient1-step11931-tokens50B"
SEED = 20260921


def main():
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B", revision=REV)
    n_tok = lambda s: len(tok(s)["input_ids"])          # same call the trainer makes (default special-token handling)
    rows = [json.loads(l) for l in open(GSM, encoding="utf-8") if l.strip()]
    _, _, _, eval_ids = split_rows(rows, 0.8, 777)
    idx = list(range(len(rows))); random.Random(777).shuffle(idx); train_pos = idx[:800]

    def full_prompt(ex):
        q, _, instr = fields_for_mode(ex, "original"); return f"{instr}\n\nQuestion: {q}\nAnswer:"
    target = {p: n_tok(full_prompt(rows[p])) for p in train_pos}
    # "carries no information": no template words, no digits, no '#'. Tokens are otherwise uniform over the vocabulary.
    BAN = ("answer", "question", "solve", "math", "problem", "reason", "number", "final", "#")
    clean = lambda s: bool(s) and "\ufffd" not in s and s.isprintable() and not any(c.isdigit() for c in s) and not any(b in s.lower() for b in BAN)
    special = set(tok.all_special_ids)
    ok_ids = [i for i in range(tok.vocab_size) if i not in special and clean(tok.decode([i])) and tok.decode([i]).strip()]
    rng = random.Random(SEED); raw = {}
    for p in train_pos:
        ids, best = [], None
        for _ in range(4000):                       # grow / backtrack token by token until the re-tokenized length is exact
            s = tok.decode(ids).strip(); n = n_tok(s) if s else 0
            if s and clean(s) and (best is None or abs(n - target[p]) < best[0]):
                best = (abs(n - target[p]), s)
            if best and best[0] == 0:
                break
            if n < target[p]:
                ids.append(rng.choice(ok_ids))
            else:
                ids = ids[:-rng.randint(1, 2)]
        raw[p] = best[1]
    out_rows = []
    for p, r in enumerate(rows):
        if p in raw:
            r = dict(r); r["raw_prompt"] = raw[p]; r["orig_question"] = raw[p]; r["orig_answer"] = "#### -999999"
        out_rows.append(r)
    path = MS / "data" / "randq_E_rawtok.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    chk = [json.loads(l) for l in open(path, encoding="utf-8")]
    _, ev_rows, tr2, ev2 = split_rows(chk, 0.8, 777)
    diffs = [n_tok(raw[p]) - target[p] for p in train_pos]; lens = [n_tok(raw[p]) for p in train_pos]; tl = list(target.values())
    q = lambda xs, a: sorted(xs)[int(a * (len(xs) - 1))]
    rep = dict(file=str(path.relative_to(MS)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seed=SEED,
               eval_ids_identical_to_gsm=(ev2 == eval_ids), eval_rows_with_raw_prompt=sum("raw_prompt" in r for r in ev_rows),
               train_rows_with_raw_prompt=sum("raw_prompt" in r for r in chk) , vocab_pool=len(ok_ids),
               gsm_full_prompt_tokens=dict(min=min(tl), median=q(tl, .5), max=max(tl)), raw_prompt_tokens=dict(min=min(lens), median=q(lens, .5), max=max(lens)),
               diff=dict(exact=sum(d == 0 for d in diffs), within1=sum(abs(d) <= 1 for d in diffs), mean_abs=round(statistics.mean(map(abs, diffs)), 3), max_abs=max(map(abs, diffs))),
               template_words_in_raw_prompts=sum(not clean(raw[p]) for p in train_pos), prompts_with_digits=sum(any(c.isdigit() for c in raw[p]) for p in train_pos),
               examples=[raw[p] for p in train_pos[:2]])
    json.dump(rep, open(HERE / "dataset_E_report.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
