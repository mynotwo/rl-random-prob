#!/usr/bin/env python3
# make_synth_gsm_ultralight.py
# Fast synthetic GSM-style generator:
# - Python renders skeleton + CoT (no LLM)
# - Optional batched paraphrasing of questions with a small model (FLAN-T5-small by default)
# - Jabberwocky nouns + number masking for "processed" views

from __future__ import annotations
# --- portability shim (added when consolidating 138/141) ---
# Override with the PROBE_REPO / PROBE_OUT environment variables.
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("PROBE_REPO", _Path(__file__).resolve().parents[1]))
PROBE_OUT = _Path(_os.environ.get("PROBE_OUT", REPO_ROOT / "experiments"))
# ----------------------------------------------------------
import os, re, json, math, random, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or ())

        def update(self, n=1):
            pass

    def tqdm(iterable=None, **kwargs):
        return _TqdmFallback(iterable, **kwargs)

# ------------------ Difficulty & skeleton config ------------------

@dataclass
class Difficulty:
    add: Tuple[int,int]
    sub: Tuple[int,int]
    mult: Tuple[int,int]
    rate: Tuple[int,int]
    sessions: Tuple[int,int]
    boxes: Tuple[int,int]
    perbox: Tuple[int,int]
    extra: Tuple[int,int]
    composite_prob: float

DIFFICULTY = {
    "easy":   Difficulty(add=(3,20),  sub=(1,10),  mult=(2,5), rate=(10,40), sessions=(2,5), boxes=(2,6), perbox=(3,10), extra=(2,8),  composite_prob=0.20),
    "medium": Difficulty(add=(10,60), sub=(2,20),  mult=(2,7), rate=(15,60), sessions=(2,8), boxes=(3,9), perbox=(4,15), extra=(3,15), composite_prob=0.35),
    "hard":   Difficulty(add=(20,120),sub=(3,40),  mult=(2,9), rate=(20,80), sessions=(3,10),boxes=(3,12),perbox=(6,20),extra=(5,25), composite_prob=0.50),
}

NAMES = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Heidi","Ivan","Judy","Mallory","Niaj",
         "Olivia","Peggy","Rupert","Sybil","Trent","Uma","Victor","Wendy","Xavier","Yvonne","Zara"]
OBJECTS = ["apple","orange","banana","book","pencil","sticker","marble","cookie","notebook","pen","toy",
           "ball","bottle","sandwich","cupcake","page","coin"]
VEHICLES = ["bus","train","car","bicycle","tram","truck"]
TASKS = ["reading","running","painting","coding","practicing piano","studying"]

def rint(a,b): return random.randint(a,b)

# Skeletons: return (meta, base_question, gold_cot_with_final)
def sk_add_sub(diff: Difficulty):
    name = random.choice(NAMES); obj = random.choice(OBJECTS)
    a, b = rint(*diff.add), rint(*diff.add)
    c = rint(1, max(1, (a+b)//3))
    total = a + b - c
    q = f"{name} had {a} {obj}s, then bought {b} more, and gave away {c}. How many {obj}s now?"
    cot = (f"Let's reason step by step.\n"
           f"Start with {a}. After buying: {a}+{b}={a+b}.\n"
           f"After giving away {c}: {a+b}-{c}={total}.\n"
           f"#### {total}")
    return {"answer": total}, q, cot

def sk_equal_groups(diff: Difficulty):
    name = random.choice(NAMES); obj = random.choice(OBJECTS)
    boxes, per = rint(*diff.boxes), rint(*diff.perbox)
    used = rint(1, max(1, per-1))
    total = boxes*per - used
    q = f"There are {boxes} boxes with {per} {obj}s each. {name} uses {used}. How many remain?"
    cot = (f"Boxes total: {boxes}×{per}={boxes*per}.\n"
           f"Remaining: {boxes*per}-{used}={total}.\n"
           f"#### {total}")
    return {"answer": total}, q, cot

def sk_linear_eq(diff: Difficulty):
    name = random.choice(NAMES); obj = random.choice(OBJECTS)
    times = rint(2, max(3,diff.mult[1])); x = rint(3,20)
    more = x*(times-1)
    q = f"{name} has {more} more {obj}s than a friend and has {times} times as many as the friend. How many does the friend have?"
    cot = (f"Let x be friend's count.\n"
           f"{times}x = x + {more} ⇒ ({times-1})x={more} ⇒ x={more}/{times-1}={x}.\n"
           f"#### {x}")
    return {"answer": x}, q, cot

def sk_money_change(diff: Difficulty):
    name = random.choice(NAMES); item = random.choice(OBJECTS)
    price, qty = rint(2,15), rint(2,8)
    pay = rint(price*qty, price*qty + diff.add[1])
    change = pay - price*qty
    q = f"{name} buys {qty} {item}s at ${price} each and pays ${pay}. How much change does {name} get?"
    cot = (f"Cost: {qty}×{price}={qty*price}.\n"
           f"Change: {pay}-{qty*price}={change}.\n"
           f"#### {change}")
    return {"answer": change}, q, cot

def sk_time_accum(diff: Difficulty):
    name = random.choice(NAMES); task = random.choice(TASKS)
    per, sessions, extra = rint(*diff.add), rint(*diff.sessions), rint(*diff.extra)
    total = per*sessions + extra
    q = f"{name} spends {per} minutes {task} for {sessions} sessions, plus a {extra}-minute break. How many minutes total?"
    cot = (f"{per}×{sessions}={per*sessions}.\n"
           f"Add break: {per*sessions}+{extra}={total}.\n"
           f"#### {total}")
    return {"answer": total}, q, cot

def sk_distance(diff: Difficulty):
    name = random.choice(NAMES); vehicle = random.choice(VEHICLES)
    speed, hours = rint(*diff.rate), rint(1,8)
    dist = speed*hours
    q = f"{name} takes a {vehicle} at {speed} mph for {hours} hours. How many miles is that?"
    cot = (f"Distance = speed×time = {speed}×{hours}={dist}.\n"
           f"#### {dist}")
    return {"answer": dist}, q, cot

def sk_share_remainder(diff: Difficulty):
    obj = random.choice(OBJECTS)
    total, kids = rint(*diff.add), rint(3,9)
    each, rem = total//kids, total%kids
    q = f"{total} {obj}s are shared equally among {kids} children. How many per child and how many remain?"
    cot = (f"{total}÷{kids} gives {each} each, remainder {rem}.\n"
           f"#### {each} remainder {rem}")
    return {"answer": f"{each} remainder {rem}"}, q, cot

def sk_two_stage(diff: Difficulty):
    name = random.choice(NAMES); obj = random.choice(OBJECTS)
    start, add, mult, give = rint(*diff.add), rint(*diff.add), rint(*diff.mult), rint(*diff.sub)
    after = (start + add) * mult
    final = after - give
    q = f"{name} starts with {start} {obj}s, gets {add} more, amount is multiplied by {mult}, then gives away {give}. How many now?"
    cot = (f"Start {start}; add: {start}+{add}={start+add}; multiply: {start+add}×{mult}={after}; give away {give}: {after}-{give}={final}.\n"
           f"#### {final}")
    return {"answer": final}, q, cot

def sk_pure_arith(diff: Difficulty):
    """Pure arithmetic chain, no word-problem wrapping."""
    a, b = rint(*diff.add), rint(*diff.mult)
    s = a + b
    result = s * b
    q = f"Compute: {a} + {b}, then multiply the result by {b}."
    cot = (f"{a}+{b}={s}. {s}×{b}={result}.\n"
           f"#### {result}")
    return {"answer": result}, q, cot

def sk_pure_structure(diff: Difficulty):
    """Multi-step logic with trivial numbers (1-3), tests reasoning structure not arithmetic."""
    name = random.choice(NAMES)
    has = rint(1, 3); gets = rint(1, 2)
    gives = rint(0, max(0, has + gets - 1))
    then_gets = rint(1, 2)
    final = has + gets - gives + then_gets
    q = (f"{name} has {has}, gets {gets} more, gives away {gives}, "
         f"then receives {then_gets}. How many now?")
    cot = (f"Start: {has}. After getting: {has}+{gets}={has+gets}. "
           f"After giving: {has+gets}-{gives}={has+gets-gives}. "
           f"After receiving: {has+gets-gives}+{then_gets}={final}.\n"
           f"#### {final}")
    return {"answer": final}, q, cot

SKELETONS = [sk_add_sub, sk_equal_groups, sk_linear_eq, sk_money_change, sk_time_accum, sk_distance, sk_share_remainder, sk_two_stage]
SKELETONS_PURE_ARITH = [sk_pure_arith]
SKELETONS_PURE_STRUCTURE = [sk_pure_structure]
SKELETON_SETS = {
    "all": SKELETONS,
    "pure_arith": SKELETONS_PURE_ARITH,
    "pure_structure": SKELETONS_PURE_STRUCTURE,
}

# ------------------ Jabberwocky + number masking ------------------

_CONS = ["b","br","bl","c","ch","cl","cr","d","dr","f","fl","fr","g","gl","gr","h","j","k","kl","kr","l","m","n",
         "p","pl","pr","qu","r","s","sc","sh","sk","sl","sm","sn","sp","st","str","sw","t","tr","v","w","wh","y","z","zr","zh"]
_VOWS = ["a","e","i","o","u","aa","ai","au","ea","ee","ei","ia","io","oa","oi","oo","ou","ua","ue"]
_TAIL = ["b","c","ck","d","f","ff","g","gh","k","kk","l","ll","lt","lp","m","mm","n","nn","nt","nk","p","pp","q","r","rr","rk","rs","rt","s","ss","st","t","tt","th","v","w","x","y","z","zz","sh","th","ng"]
def make_nonsense():
    syls = random.choice([2,2,3,3,3,4])
    parts=[]; 
    for _ in range(syls):
        parts += [random.choice(_CONS), random.choice(_VOWS)]
        if random.random()<0.5: parts.append(random.choice(_TAIL))
    token = re.sub(r"[^a-z]","", "".join(parts).lower())
    return token if len(token)>=4 else token + random.choice(["ix","um","or","an","eth"])

BASE_NOUNS = ["apple","orange","banana","book","pencil","sticker","marble","cookie","notebook","pen","toy","ball",
              "bottle","sandwich","cupcake","page","coin","friend","neighbor","student","teacher","store","market","bakery",
              "library","park","car","bus","train","bicycle","tram","truck","ticket","minute","hour","dollar","bag","chair",
              "table","crayon","eraser","ruler","marker","sheet","problem","question","answer","class","game","team","player","kid","child"]
def plural(w:str)->str:
    if w.endswith("y"): return w[:-1]+"ies"
    if w.endswith(("s","ch","sh")): return w+"es"
    return w+"s"
NOUN_SET = set(BASE_NOUNS) | {plural(w) for w in BASE_NOUNS}
NOUN_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, NOUN_SET), key=len, reverse=True)) + r")\b", re.I)

def jabberwocky_pair(question: str, answer: str) -> tuple[str,str]:
    mapping: Dict[str,str] = {}
    def repl(m):
        w = m.group(0); key = w.lower()
        if key not in mapping: mapping[key]=make_nonsense()
        rep = mapping[key]
        return rep.capitalize() if w[0].isupper() else rep
    jq = NOUN_RE.sub(repl, question)
    ja = NOUN_RE.sub(repl, answer)
    # replace capitalized names (not at sentence start)
    def replace_names(text: str) -> str:
        def sub_name(m):
            w=m.group(0)
            if w=="I": return w
            k=w.lower()
            if k not in mapping: mapping[k]=make_nonsense()
            return mapping[k].capitalize()
        return re.sub(r"(?<![\.\!\?]\s)(\b[A-Z][a-z]+\b)", sub_name, text)
    return replace_names(jq), replace_names(ja)

def build_num_masker(token="<NUM>"):
    words = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
             "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million").split()
    re_words = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)
    re_nums  = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?")
    def mask_numbers(text: str) -> str:
        text = re_nums.sub(token, text)
        text = re_words.sub(token, text)
        return text
    return mask_numbers

# ------------------ Lightweight paraphrasing (batched) ------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:/\d+)?")

def numbers_in(s: str) -> List[str]:
    return _NUM_RE.findall(s)

def batched_paraphrase_seq2seq(
    texts: List[str],
    tok,
    mdl,
    device_map: str | dict = "auto",
    batch_size: int = 64,
    num_return_sequences: int = 2,
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    top_p: float = 0.95,
    task_prefix: str = "Paraphrase the following while preserving ALL numbers and semantics. Output only the rewritten text:\n\n",
) -> List[List[str]]:
    """
    Paraphrase arbitrary short/medium texts with a small seq2seq model.
    Returns list-of-lists (candidates per input).
    """
    prompts = [task_prefix + t for t in texts]
    outs: List[List[str]] = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=384, return_tensors="pt")
        dev = getattr(mdl, "device", None) or next(mdl.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        gen = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch)):
            outs.append(dec[j*num_return_sequences:(j+1)*num_return_sequences])
    return outs


# ------------------ Main generator ------------------

def render_batch(n: int, diff: Difficulty, skeleton_set: str = "all") -> List[Dict[str,str]]:
    """Render a batch of base items without paraphrasing (fast)."""
    skel_list = SKELETON_SETS.get(skeleton_set, SKELETONS)
    items = []
    for _ in range(n):
        if skeleton_set == "all":
            fn = (sk_two_stage if random.random()<diff.composite_prob else random.choice(skel_list))
        else:
            fn = random.choice(skel_list)
        meta, q, cot = fn(diff)
        items.append({"orig_question": q, "orig_answer": cot})
    return items

_FINAL_RE = re.compile(r"^\s*####\s*(.+?)\s*$")

def split_reasoning_and_final(answer_text: str) -> tuple[str, str]:
    """
    Splits CoT into (reasoning_block, final_line_including_hashes).
    If no final line found, returns (whole_text, "").
    """
    lines = [ln for ln in answer_text.strip().splitlines()]
    for i in range(len(lines)-1, -1, -1):
        if _FINAL_RE.match(lines[i]):
            reasoning = "\n".join(lines[:i]).strip()
            final = lines[i].strip()
            return reasoning, final
    return answer_text.strip(), ""  # fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(REPO_ROOT) + "/src/data/synthetic_gsm.jsonl")
    ap.add_argument("--num-examples", type=int, default=10000)
    ap.add_argument("--difficulty", type=str, choices=["easy","medium","hard"], default="medium")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skeleton_filter", type=str, default="all",
                    choices=list(SKELETON_SETS.keys()),
                    help="Which skeleton set to use: all (default), pure_arith, pure_structure")

    # Paraphrasing controls
    ap.add_argument("--use_paraphrase_questions", action="store_true",
                    help="Enable small-model batched paraphrasing for question diversity")
    ap.add_argument("--paraphrase_model", type=str, default="google/flan-t5-small")
    ap.add_argument("--paraphrase_batch", type=int, default=1024)
    ap.add_argument("--paraphrase_n", type=int, default=2, help="num_return_sequences per question")
    ap.add_argument("--paraphrase_keep_prob", type=float, default=0.7, help="probability to replace question with a valid paraphrase")
    ap.add_argument("--device_map", type=str, default="auto")

    # Batching
    ap.add_argument("--skeleton_batch", type=int, default=1024, help="how many base items to render per outer loop")

    ap.add_argument("--use_paraphrase_answers", action="store_true",
                help="Enable batched paraphrasing of CoT reasoning (final #### line kept as-is).")
    ap.add_argument("--ans_paraphrase_batch", type=int, default=1024)
    ap.add_argument("--ans_paraphrase_n", type=int, default=2,
                    help="num_return_sequences per reasoning text")
    ap.add_argument("--ans_paraphrase_keep_prob", type=float, default=0.7,
                    help="probability to replace reasoning with a valid paraphrase")
    ap.add_argument("--ans_max_new_tokens", type=int, default=256)

    args = ap.parse_args()


    random.seed(args.seed)

    diff = DIFFICULTY[args.difficulty]
    mask_numbers = build_num_masker("<NUM>")

    tok = None
    mdl = None
    if args.use_paraphrase_questions or args.use_paraphrase_answers:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tok = AutoTokenizer.from_pretrained(args.paraphrase_model)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            args.paraphrase_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map=args.device_map
        )

    pbar = tqdm(total=args.num_examples)
    written = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        while written < args.num_examples:
            need = min(args.num_examples - written, args.skeleton_batch)
            batch = render_batch(need, diff, skeleton_set=args.skeleton_filter)

            # optional paraphrase (batched small model, fast)
            if args.use_paraphrase_questions:
                assert tok is not None and mdl is not None
                qs = [it["orig_question"] for it in batch]
                cand_lists = batched_paraphrase_seq2seq(
                    qs,
                    tok,
                    mdl,
                    batch_size=args.paraphrase_batch,
                    num_return_sequences=args.paraphrase_n,
                    max_new_tokens=128,
                    temperature=0.7, top_p=0.95,
                    task_prefix=("Paraphrase the following math word problem into a single paragraph, "
                                "preserving ALL numbers, names, and semantics. "
                                "Do NOT include any solution.\n\n"),
                )
                for it, cands in zip(batch, cand_lists):
                    base_nums = numbers_in(it["orig_question"])
                    chosen = None
                    for c in cands:
                        c = c.strip()
                        if not c:
                            continue
                        if numbers_in(c) == base_nums and random.random() < args.paraphrase_keep_prob:
                            chosen = c
                            break
                    if chosen:
                        it["orig_question"] = chosen

            
            if args.use_paraphrase_answers:
                assert tok is not None and mdl is not None
                # 1) Split reasoning/final, keep arrays aligned with `batch`
                reasonings: List[str] = []
                finals: List[str] = []
                for it in batch:
                    r, fin = split_reasoning_and_final(it["orig_answer"])
                    reasonings.append(r)
                    finals.append(fin)

                # 2) Paraphrase reasoning texts in batch (small seq2seq)
                cand_lists = batched_paraphrase_seq2seq(
                    reasonings,
                    tok,
                    mdl,
                    batch_size=args.ans_paraphrase_batch,
                    num_return_sequences=args.ans_paraphrase_n,
                    max_new_tokens=args.ans_max_new_tokens,
                    temperature=0.7, top_p=0.95,
                    task_prefix=(
                        "Paraphrase the following chain-of-thought reasoning into clear, concise steps, "
                        "while preserving ALL numbers and the step semantics. "
                        "Do NOT add or remove steps and do NOT include the final answer line. "
                        "Output only the rewritten reasoning:\n\n"
                    ),
                )

                # 3) Select a candidate that preserves the multiset of numbers; re-attach final line
                for it, base_r, base_fin, cands in zip(batch, reasonings, finals, cand_lists):
                    base_nums = numbers_in(base_r)
                    chosen = None
                    for c in cands:
                        c = c.strip()
                        if not c:
                            continue
                        if numbers_in(c) == base_nums and random.random() < args.ans_paraphrase_keep_prob:
                            chosen = c
                            break
                    if chosen:
                        # Restore as: paraphrased reasoning + original final #### line (unchanged)
                        if base_fin:
                            it["orig_answer"] = (chosen.rstrip() + "\n" + base_fin).strip()
                        else:
                            # no final line found; just replace reasoning
                            it["orig_answer"] = chosen


            # post-process: jabberwocky + masking
            for it in batch:
                pq, pa = jabberwocky_pair(it["orig_question"], it["orig_answer"])
                it_out = {
                    "orig_question": it["orig_question"],
                    "processed_question_jm": mask_numbers(pq),
                    "processed_question_m": mask_numbers(it["orig_question"]),
                    "orig_answer": it["orig_answer"],
                    "processed_answer_jm": mask_numbers(pa),
                    "processed_answer_m": mask_numbers(it["orig_answer"]),
                }
                f.write(json.dumps(it_out, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)
                if written >= args.num_examples:
                    break

            print(f"[progress] {written}/{args.num_examples} written")

    print(f"Done. Wrote {written} examples to {args.out}")

if __name__ == "__main__":
    main()

# python make_synth_gsm.py --num-examples 10000
# python make_synth_gsm.py --num-examples 10000 --use_paraphrase_questions --use_paraphrase_answers
