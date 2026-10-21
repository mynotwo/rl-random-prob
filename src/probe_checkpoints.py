#!/usr/bin/env python3
# probe_checkpoints.py
from __future__ import annotations
# --- portability shim (added when consolidating 138/141) ---
# Override with the PROBE_REPO / PROBE_OUT environment variables.
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("PROBE_REPO", _Path(__file__).resolve().parents[1]))
PROBE_OUT = _Path(_os.environ.get("PROBE_OUT", REPO_ROOT / "experiments"))
# ----------------------------------------------------------
import os, json, math, re, random, argparse, time
from typing import List, Dict, Any, Optional
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# ---------------------------
# utils: reproducibility
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# data loading
# ---------------------------
def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows

def split_train_eval(rows: List[Dict[str, str]], train_frac: float, seed: int):
    idx = list(range(len(rows)))
    rnd = random.Random(seed); rnd.shuffle(idx)
    k = int(len(rows) * train_frac)
    tr = [rows[i] for i in idx[:k]]
    ev = [rows[i] for i in idx[k:]]
    return tr, ev


# ---------------------------
# prompting and formatting
# ---------------------------
INSTR_ORIG = "Solve the math problem. Show your reasoning and put the final answer on a line like: #### <number>."
INSTR_PROC = "Solve the problem using placeholders. Show reasoning. Put the final answer on a line like: #### <number>."

def fields_for_mode(ex: Dict[str,str], mode: str) -> tuple[str, str, str]:
    if mode == "original":
        q = ex["orig_question"].strip()
        a = ex["orig_answer"].strip()
        instr = INSTR_ORIG
    elif mode == "processed_jm":
        q = ex["processed_question_jm"].strip()
        a = ex["processed_answer_jm"].strip()
        instr = INSTR_PROC
    elif mode == "processed_m":
        q = ex["processed_question_m"].strip()
        a = ex["processed_answer_m"].strip()
        instr = INSTR_PROC
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return q, a, instr

def build_train_text(ex: Dict[str,str], mode: str) -> str:
    q, a, instr = fields_for_mode(ex, mode)
    return f"{instr}\n\nQuestion: {q}\nAnswer: {a}"

def build_eval_prompt(ex: Dict[str,str]) -> str:
    q = ex["orig_question"].strip()
    return f"{INSTR_ORIG}\n\nQuestion: {q}\nAnswer:"

# ---------------------------
# answer extraction (GSM-style)
# ---------------------------
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:/\d+)?")

def _to_float(tok: str) -> Optional[float]:
    try:
        if "/" in tok and tok.count("/") == 1:
            a,b = tok.split("/")
            return float(a)/float(b)
        return float(tok)
    except Exception:
        return None

def extract_final_number(text: str) -> Optional[float]:
    if not text: return None
    m = re.search(r"####\s*(" + _NUM.pattern + r")", text)
    if m: return _to_float(m.group(1))
    m = re.search(r"<<[^<>]*=\s*(" + _NUM.pattern + r")\s*>>", text)
    if m: return _to_float(m.group(1))
    m = re.search(r"(?:answer\s*(?:is|:)\s*)(" + _NUM.pattern + r")", text, flags=re.I)
    if m: return _to_float(m.group(1))
    nums = _NUM.findall(text)
    return _to_float(nums[-1]) if nums else None

def extract_strict_hash_number(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"^\s*####\s*(" + _NUM.pattern + r")\s*$", text, flags=re.M)
    return _to_float(m.group(1)) if m else None

# ---------------------------
# evaluation (generation-based)
# ---------------------------
@torch.no_grad()
def batched_generate(tokenizer, model, prompts: List[str],
                     max_new_tokens=128, batch_size=4,
                     temperature=0.0, top_p=1.0, num_return_sequences=1) -> List[str] | List[List[str]]:
    results = []
    pad_side_orig = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            dev = next(model.parameters()).device
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # do_sample=(temperature > 0.0),
                # temperature=temperature,
                # top_p=top_p,
                # num_return_sequences=num_return_sequences,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            # strip prompts
            seq_len = enc["input_ids"].shape[1]
            dec = tokenizer.batch_decode(out[:, seq_len:], skip_special_tokens=True)
            if num_return_sequences == 1:
                results.extend(dec)
            else:
                for j in range(len(batch)):
                    results.append(dec[j*num_return_sequences:(j+1)*num_return_sequences])
    finally:
        tokenizer.padding_side = pad_side_orig
    return results

def evaluate_accuracy_on_original(tokenizer, model, eval_rows: List[Dict[str,str]],
                                  max_new_tokens=128, batch_size=4, temperature=0.0, top_p=1.0,
                                  num_return_sequences=1, majority_vote=False) -> float:
    prompts = [build_eval_prompt(ex) for ex in eval_rows]
    print('begin evaluation ...')
    gens = batched_generate(tokenizer, model, prompts, max_new_tokens, batch_size, temperature, top_p, num_return_sequences)
    correct = 0
    if num_return_sequences == 1 or not majority_vote:
        for ex, pred in zip(eval_rows, gens):  # gens: List[str]
            gold = extract_final_number(ex["orig_answer"])
            got = extract_final_number(pred)
            ok = (gold is not None and got is not None and math.isclose(got, gold, rel_tol=0.0, abs_tol=1e-6))
            correct += int(ok)
    else:
        from collections import Counter
        for ex, pred_list in zip(eval_rows, gens):  # gens: List[List[str]]
            gold = extract_final_number(ex["orig_answer"])
            nums = [extract_final_number(s) for s in pred_list]
            nums = [n for n in nums if n is not None]
            if not nums or gold is None:
                continue
            c = Counter(nums); vote = max(c.items(), key=lambda kv: kv[1])[0]
            if math.isclose(vote, gold, rel_tol=0.0, abs_tol=1e-6):
                correct += 1
    return correct / max(1, len(eval_rows))

# ---------------------------
# callback to eval each epoch (rank 0 only)
# ---------------------------
class EvalEachEpoch(TrainerCallback):
    def __init__(self, tokenizer, eval_rows, out_path: str,
                 gen_cfg: Dict[str,Any]):
        self.tok = tokenizer
        self.rows = eval_rows
        self.out_path = out_path
        self.gen_cfg = gen_cfg
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def on_train_begin(self, args, state, control, **kw):
        if int(os.environ.get("RANK","0")) != 0: return
        model = kw["model"]
        # pre-train eval
        restore = getattr(model.config, "use_cache", None)
        if hasattr(model.config, "use_cache"): model.config.use_cache = True
        acc = evaluate_accuracy_on_original(self.tok, model, self.rows, **self.gen_cfg)
        if restore is not None: model.config.use_cache = restore
        rec = {"phase":"before", "epoch":0, "step":int(state.global_step), "accuracy":acc, "ts":time.time()}
        with open(self.out_path, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
        print(f"[Eval before] acc={acc:.3f}")

    def on_epoch_end(self, args, state, control, **kw):
        if int(os.environ.get("RANK","0")) != 0: return
        model = kw["model"]
        restore = getattr(model.config, "use_cache", None)
        if hasattr(model.config, "use_cache"): model.config.use_cache = True
        acc = evaluate_accuracy_on_original(self.tok, model, self.rows, **self.gen_cfg)
        if restore is not None: model.config.use_cache = restore
        rec = {"phase":"epoch_end", "epoch":int(state.epoch or 0), "step":int(state.global_step), "accuracy":acc, "ts":time.time()}
        with open(self.out_path, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
        print(f"[Eval epoch={int(state.epoch or 0)}] acc={acc:.3f}")

def _right_pad_batch(tokenizer, features):
    """
    Manually right-pad input_ids, attention_mask, and labels to a common length.
    - input_ids: pad with pad_token_id (or eos_token_id if pad is None)
    - attention_mask: pad with 0
    - labels: pad with -100 (ignored by CE)
    Returns a dict of torch.LongTensor.
    """
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # fall back to EOS
        pad_id = tokenizer.eos_token_id
    assert pad_id is not None, "Tokenizer must have pad_token_id or eos_token_id."

    max_len = max(len(f["input_ids"]) for f in features)

    input_ids, attn, labels = [], [], []
    for f in features:
        L = len(f["input_ids"])
        pad_len = max_len - L

        input_ids.append(f["input_ids"] + [pad_id] * pad_len)
        attn.append(f["attention_mask"] + [0] * pad_len)

        lab = f["labels"] + [-100] * pad_len
        labels.append(lab)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return batch


def _mk_ce_items(rows, mode, tokenizer, max_len: int) -> List[Dict[str, List[int]]]:
    """
    Build items with input_ids, attention_mask, labels so that loss is
    computed only on the ANSWER tokens given the QUESTION prompt.
    mode: "original", "processed_jm", or "processed_m".
    """
    items = []
    for ex in rows:
        q, a, instr = fields_for_mode(ex, mode)

        prompt = f"{instr}\n\nQuestion: {q}\nAnswer:"
        # Encode separately to know the split point
        enc_p = tokenizer(prompt, add_special_tokens=False)
        enc_a = tokenizer(a, add_special_tokens=False)
        input_ids = enc_p["input_ids"] + enc_a["input_ids"]
        # Truncate from the left if too long (keep the answer tail)
        if len(input_ids) > max_len:
            # compute how many prompt tokens we can keep (always keep full answer if possible)
            keep_from = max(0, len(input_ids) - max_len)
            # shift the split index accordingly
            split = max(0, len(enc_p["input_ids"]) - keep_from)
            input_ids = input_ids[keep_from:]
        else:
            split = len(enc_p["input_ids"])
        labels = [-100] * split + input_ids[split:]  # CE only on answer tokens
        attention_mask = [1] * len(input_ids)
        items.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        })
    return items

@torch.no_grad()
def evaluate_ce_over_test(tokenizer, model, rows, eval_mode: str, max_len: int = 1024, batch_size: int = 8) -> Dict[str, float]:
    """
    Compute average cross-entropy (token NLL) over the test set answers.
    eval_mode: "original" => use (orig_question, orig_answer)
               "processed_jm" => use jabberwocky+masked fields
               "processed_m" => use masked-number fields
    Returns: {"nll": float, "ppl": float}
    """
    model.eval()
    dev = next(model.parameters()).device
    items = _mk_ce_items(rows, eval_mode, tokenizer, max_len)
    # Batch
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch = _right_pad_batch(tokenizer, batch)
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = model(**batch)
        # out.loss is already mean over non-masked tokens in the batch
        # To be precise by token-count, recompute per-token loss:
        logits = out.logits  # [B,T,V]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        # mask
        mask = (shift_labels != -100)
        # compute per-token nll
        logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        gold = shift_labels.masked_fill(~mask, 0)
        nll_tok = -logprobs.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
        nll_tok = nll_tok.masked_select(mask)
        total_loss += nll_tok.sum().item()
        total_tokens += mask.sum().item()
    avg_nll = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_nll) if avg_nll < 100 else float("inf")
    return {"nll": avg_nll, "ppl": ppl}


# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, default=str(REPO_ROOT) + "/src/data/synthetic_gsm.jsonl")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--base_model", type=str, default="allenai/OLMo-2-1124-7B")
    ap.add_argument("--revision", type=str, default=None, help="HF revision for the checkpoint (e.g., step tag)")
    ap.add_argument("--train_mode", type=str, choices=["original","processed_jm", "processed_m"], default="processed_m")
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--train_percent", type=float, default=1.0, help="subset of the *train* split to use (0< p <=1)")
    ap.add_argument("--seed", type=int, default=42)

    # training args
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--logging_steps", type=int, default=50)

    # --- LoRA options ---
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    # Common OLMo-2 target modules (LLaMA-style names). You can override if your checkpoint differs.
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="auto",
        help="Comma-separated module names to adapt, or 'auto' for known OLMo/Pythia defaults"
    )
    ap.add_argument("--save_adapters_only", action="store_true", help="Save only LoRA adapters at the end")

    # generation eval args
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--gen_max_new_tokens", type=int, default=128)
    # ap.add_argument("--gen_temperature", type=float, default=0.0)
    # ap.add_argument("--gen_top_p", type=float, default=1.0)
    # ap.add_argument("--gen_num_return_sequences", type=int, default=1)
    # ap.add_argument("--gen_majority_vote", action="store_true")
    ap.add_argument("--eval_metric", type=str, choices=["accuracy", "cross_entropy"], default="accuracy",
                help="Evaluation metric. For MDL use 'cross_entropy'.")
    ap.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for CE eval.")


    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # load data & split
    rows = load_jsonl(args.dataset_path)
    train_rows, eval_rows = split_train_eval(rows, args.train_frac, args.seed)

    # subset by percent
    if not (0 < args.train_percent <= 1.0):
        raise ValueError("--train_percent must be in (0,1].")
    k = max(1, int(len(train_rows) * args.train_percent))
    train_rows = train_rows[:k]

    if args.eval_metric == "cross_entropy":
        eval_mode_for_ce = args.train_mode  # "processed" or "original"
    else:
        eval_mode_for_ce = "original"  # unused in accuracy mode

    # build training texts
    train_texts = [build_train_text(ex, args.train_mode) for ex in train_rows]
    print(f"Training examples: {len(train_texts)} (mode={args.train_mode}) | Eval examples: {len(eval_rows)}")

    # tokenizer + model (no gradient checkpointing; no device_map)
    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    if args.revision and '-step0-' in args.revision:
        cfg = AutoConfig.from_pretrained(args.base_model, revision=args.revision)
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            revision=args.revision,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            # device_map='auto'
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # safer for training
    # NEW: apply LoRA if requested
    if args.use_lora:
        if args.lora_target_modules.strip().lower() != "auto":
            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        elif 'OLMo' in args.base_model:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        elif 'pythia' in args.base_model.lower():
            target_modules = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']
        elif 'qwen' in args.base_model.lower():
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        else:
            target_modules = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']
        if not target_modules:
            raise ValueError("--lora_target_modules must include at least one module name.")
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    base_model_name = args.base_model.split('/')[-1]

    # datasets & collator
    train_ds = Dataset.from_dict({"text": train_texts})
    def tokenize(examples):
        return tok(examples["text"], truncation=True, max_length=args.max_seq_length)
    tokenized = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # training args
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="no",    # we do our own generation-based eval in callback
        report_to=[],
        bf16=torch.cuda.is_available(),  # H100-friendly
        ddp_find_unused_parameters=False, # makes DDP happier with adapters
    )

    # evaluation callback
    gen_cfg = dict(
        max_new_tokens=args.gen_max_new_tokens,
        batch_size=args.gen_batch_size,
        # temperature=args.gen_temperature,
        # top_p=args.gen_top_p,
        # num_return_sequences=args.gen_num_return_sequences,
        # majority_vote=args.gen_majority_vote,
    )
    revision_name = args.revision or "main"
    metric_tag = f"-{args.eval_metric}" if args.eval_metric != "accuracy" else ""
    eval_log = os.path.join(args.output_dir, f"eval_epochs_{args.train_mode}-{base_model_name}-{revision_name}-{int(100*args.train_percent)}{metric_tag}.jsonl")
    if args.eval_metric == "cross_entropy":
        class CEEachEpoch(TrainerCallback):
            def on_train_begin(self, args_tr, state, control, **kw):
                if int(os.environ.get("RANK","0")) != 0: return
                model_ = kw["model"]
                res = evaluate_ce_over_test(tok, model_, eval_rows, eval_mode_for_ce,
                                            max_len=args.max_seq_length, batch_size=args.eval_batch_size)
                rec = {"phase":"before", "epoch":0, "step":int(state.global_step), "metric":"cross_entropy",
                    "nll":res["nll"], "ppl":res["ppl"], "ts":time.time()}
                with open(eval_log, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
                print(f"[Eval before] CE nll={res['nll']:.4f}  ppl={res['ppl']:.2f}")

            def on_epoch_end(self, args_tr, state, control, **kw):
                if int(os.environ.get("RANK","0")) != 0: return
                model_ = kw["model"]
                res = evaluate_ce_over_test(tok, model_, eval_rows, eval_mode_for_ce,
                                            max_len=args.max_seq_length, batch_size=args.eval_batch_size)
                rec = {"phase":"epoch_end", "epoch":int(state.epoch or 0), "step":int(state.global_step), "metric":"cross_entropy",
                    "nll":res["nll"], "ppl":res["ppl"], "ts":time.time()}
                with open(eval_log, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
                print(f"[Eval epoch={int(state.epoch or 0)}] CE nll={res['nll']:.4f}  ppl={res['ppl']:.2f}")

        callbacks = [CEEachEpoch()]
    else:
        # keep your existing accuracy callback
        gen_cfg = dict(
            max_new_tokens=args.gen_max_new_tokens,
            batch_size=args.gen_batch_size,
            # temperature=args.gen_temperature,
            # top_p=args.gen_top_p,
            # num_return_sequences=args.gen_num_return_sequences,
            # majority_vote=args.gen_majority_vote,
        )
        callbacks = [EvalEachEpoch(tok, eval_rows, eval_log, gen_cfg)]

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tok,
        callbacks=callbacks,
    )

    trainer.train()

    # final eval (again)
    if int(os.environ.get("RANK","0")) == 0:
        if args.eval_metric == "cross_entropy":
            res = evaluate_ce_over_test(tok, model, eval_rows, eval_mode_for_ce,
                                        max_len=args.max_seq_length, batch_size=args.eval_batch_size)
            rec = {"phase":"after", "epoch":int(args.num_train_epochs), "step":int(trainer.state.global_step),
                "metric":"cross_entropy", "nll":res["nll"], "ppl":res["ppl"], "ts":time.time()}
            with open(eval_log, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
            print(f"[Eval after] CE nll={res['nll']:.4f}  ppl={res['ppl']:.2f}")
        else:
            restore = getattr(model.config, "use_cache", None)
            if hasattr(model.config, "use_cache"): model.config.use_cache = True
            acc = evaluate_accuracy_on_original(tok, model, eval_rows, **gen_cfg)
            if restore is not None: model.config.use_cache = restore
            rec = {"phase":"after", "epoch":int(args.num_train_epochs), "step":int(trainer.state.global_step),
                "metric":"accuracy", "accuracy":acc, "ts":time.time()}
            with open(eval_log, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
            print(f"[Eval after] acc={acc:.3f}")

    
    if int(os.environ.get("RANK", "0")) == 0 and args.use_lora and args.save_adapters_only:
        adapters_dir = os.path.join(args.output_dir, "lora_adapters")
        os.makedirs(adapters_dir, exist_ok=True)
        model.save_pretrained(adapters_dir)
        tok.save_pretrained(adapters_dir)
        print(f"Saved LoRA adapters to: {adapters_dir}")

if __name__ == "__main__":
    main()
