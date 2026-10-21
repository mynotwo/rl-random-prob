#!/usr/bin/env python3
"""Random-question datasets for the "random question + random reward" control.

Each output has the same 1000 row positions as synthetic_gsm_rlvr_1000_seed13.jsonl. The 200 eval
rows (split seed 777, train_frac 0.8) are copied byte-for-byte, so the reference trainer evaluates on
exactly the GSM items of the checkpoint-curve runs (same eval_ids). Only the 800 TRAIN questions change:

  C  shuffled_gsm  the same question with its words shuffled (keeps vocabulary, numbers, length)
  A  mmlu          an unrelated natural question: MMLU stems from non-quantitative subjects, no digits
  D  randtok       uniformly random vocabulary tokens

Every replacement is matched to the token count of the GSM question it replaces (OLMo-2 tokenizer).
C keeps the original gold answer (so rollout_lenient_correct_rate says whether the model still solves
the scrambled problem); A and D get an unreachable gold (#### -999999): there is no right answer.
"""
from __future__ import annotations

import collections, hashlib, json, pathlib, random, statistics, sys

HERE = pathlib.Path(__file__).resolve().parent
MS = HERE.parent
sys.path.insert(0, str(MS / "trainer"))
from protocol import split_rows  # the reference trainer's own splitter  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

GSM = MS / "data" / "synthetic_gsm_rlvr_1000_seed13.jsonl"
REV = "stage2-ingredient1-step11931-tokens50B"
SEED = 20260920
NO_GOLD = "#### -999999"
QUANT = ("math", "algebra", "statistics", "physics", "chemistry", "econometrics", "accounting", "computer", "engineering", "astronomy")


def main():
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B", revision=REV)
    n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    rows = [json.loads(l) for l in open(GSM, encoding="utf-8") if l.strip()]
    _, _, train_ids, eval_ids = split_rows(rows, 0.8, 777)
    idx = list(range(len(rows))); random.Random(777).shuffle(idx)
    train_pos = idx[:int(len(rows) * 0.8)]
    target = {p: n_tok(rows[p]["orig_question"]) for p in train_pos}
    rng = random.Random(SEED)

    def replace(row, question, keep_gold):
        r = dict(row); r["orig_question"] = question
        if not keep_gold:
            r["orig_answer"] = NO_GOLD
        return r

    # ---- C: word shuffle, best of 60 permutations on token-count match
    c_q = {}
    for p in train_pos:
        words = rows[p]["orig_question"].split()
        best = None
        for _ in range(60):
            w = words[:]; rng.shuffle(w)
            if w == words:
                continue
            s = " ".join(w); d = abs(n_tok(s) - target[p])
            if best is None or d < best[0]:
                best = (d, s)
            if d == 0:
                break
        c_q[p] = best[1]

    # ---- A: MMLU non-quantitative, digit-free stems; nearest token count, without replacement
    mm = [json.loads(l) for l in open(MS / "data" / "mmlu_all.jsonl", encoding="utf-8") if l.strip()]
    pool = sorted({r["orig_question"].strip() for r in mm
                   if not any(q in str(r.get("subject", "")).lower() for q in QUANT)
                   and not any(ch.isdigit() for ch in r["orig_question"]) and len(r["orig_question"].strip()) > 10})
    by_len = collections.defaultdict(list)
    for q in pool:
        by_len[n_tok(q)].append(q)
    for v in by_len.values():
        rng.shuffle(v)
    a_q = {}
    for p in sorted(train_pos, key=lambda p: -target[p]):
        for d in range(0, 40):
            hit = next((l for l in (target[p] - d, target[p] + d) if by_len.get(l)), None)
            if hit is not None:
                a_q[p] = by_len[hit].pop(); break
        else:
            raise SystemExit(f"no MMLU stem near {target[p]} tokens")

    # ---- D: uniformly random vocabulary tokens that decode to clean printable text
    vocab_ids = [i for i in range(tok.vocab_size) if i not in set(tok.all_special_ids)]
    ok_ids = []
    for i in vocab_ids:
        s = tok.decode([i])
        if s and "�" not in s and s.isprintable() and s.strip():
            ok_ids.append(i)
    d_q = {}
    for p in train_pos:
        best = None
        for _ in range(200):
            k = target[p] if best is None else max(1, target[p] + (target[p] - best[2]))
            s = tok.decode([rng.choice(ok_ids) for _ in range(k)]).strip()
            n = n_tok(s); d = abs(n - target[p])
            if best is None or d < best[0]:
                best = (d, s, n)
            if d == 0:
                break
        d_q[p] = best[1]

    report = {"gsm_sha256": hashlib.sha256(GSM.read_bytes()).hexdigest(), "tokenizer_revision": REV, "seed": SEED,
              "n_train": len(train_pos), "n_eval": len(eval_ids), "arms": {}}
    for arm, name, qs, keep_gold in (("C", "shuffled_gsm", c_q, True), ("A", "mmlu", a_q, False), ("D", "randtok", d_q, False)):
        out_rows = [replace(rows[p], qs[p], keep_gold) if p in qs else rows[p] for p in range(len(rows))]
        path = MS / "data" / f"randq_{arm}_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        _, _, tr2, ev2 = split_rows([json.loads(l) for l in open(path, encoding="utf-8")], 0.8, 777)
        diffs = [n_tok(qs[p]) - target[p] for p in train_pos]
        lens = [n_tok(qs[p]) for p in train_pos]
        q = lambda xs, a: sorted(xs)[int(a * (len(xs) - 1))]
        report["arms"][arm] = dict(file=str(path.relative_to(MS)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   eval_ids_identical_to_gsm=(ev2 == eval_ids), train_ids_overlap_with_gsm=len(set(tr2) & set(train_ids)),
                                   token_len=dict(min=min(lens), p10=q(lens, .1), median=q(lens, .5), p90=q(lens, .9), max=max(lens)),
                                   diff_vs_gsm=dict(exact=sum(d == 0 for d in diffs), within1=sum(abs(d) <= 1 for d in diffs),
                                                    mean_abs=round(statistics.mean(map(abs, diffs)), 3), max_abs=max(map(abs, diffs))),
                                   examples=[qs[p] for p in train_pos[:3]])
    gl = list(target.values()); q = lambda xs, a: sorted(xs)[int(a * (len(xs) - 1))]
    report["gsm_token_len"] = dict(min=min(gl), p10=q(gl, .1), median=q(gl, .5), p90=q(gl, .9), max=max(gl))
    json.dump(report, open(HERE / "datasets_report.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
