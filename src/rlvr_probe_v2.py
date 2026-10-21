#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
# Bypass torch 2.6 requirement for .bin model loading (CVE-2025-32434)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

from probe_checkpoints import extract_final_number, extract_strict_hash_number, fields_for_mode, load_jsonl, split_train_eval

try:
    from code_eval import build_code_prompt, compute_code_rewards, extract_code_block, execute_code_with_tests
    HAS_CODE_EVAL = True
except ImportError:
    HAS_CODE_EVAL = False


INSTR_ORIG = "Solve the math problem. Show your reasoning and put the final answer on a line like: #### <number>."
INSTR_PROC = "Solve the problem using placeholders. Show reasoning. Put the final answer on a line like: #### <number>."


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt(ex: Dict[str, str], mode: str) -> str:
    q, _, instr = fields_for_mode(ex, mode)
    return f"{instr}\n\nQuestion: {q}\nAnswer:"


def gold_number(ex: Dict[str, str]) -> Optional[float]:
    return extract_final_number(ex["orig_answer"])


def behavior_features(text: str) -> Dict[str, float]:
    lowered = text.lower()
    tokens = re.findall(r"\S+", text)
    unique_tokens = set(tokens)
    step_markers = len(re.findall(r"(?:^|\n|\b)(?:step\s*\d+|first|second|third|then|therefore|so|because|start|remaining|total)\b", lowered))
    arithmetic_ops = len(re.findall(r"\d+\s*[\+\-\*×x/÷]\s*\d+", text))
    code_like = int(
        "python" in lowered
        or "```" in text
        or re.search(r"\b(for|while|def|return|print)\b", lowered) is not None
    )
    repeated_bigram = 0
    if len(tokens) >= 4:
        bigrams = list(zip(tokens, tokens[1:]))
        repeated_bigram = int(len(set(bigrams)) < len(bigrams) * 0.8)
    return {
        "response_chars": float(len(text)),
        "response_tokens": float(len(tokens)),
        "unique_token_ratio": float(len(unique_tokens) / max(1, len(tokens))),
        "step_marker_count": float(step_markers),
        "arithmetic_op_count": float(arithmetic_ops),
        "code_like": float(code_like),
        "repeated_bigram": float(repeated_bigram),
        "format_ok": float("####" in text or "\\boxed" in text),
        "extract_ok": float(extract_final_number(text) is not None),
        "strict_hash_extract_ok": float(extract_strict_hash_number(text) is not None),
    }


def target_modules_for(base_model: str, override: str) -> list[str]:
    if override.strip().lower() != "auto":
        modules = [m.strip() for m in override.split(",") if m.strip()]
    elif "OLMo" in base_model:
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "pythia" in base_model.lower():
        modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif any(k in base_model.lower() for k in ["qwen", "llama", "deepseek", "mistral", "gemma"]):
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if not modules:
        raise ValueError("--lora_target_modules must include at least one module name.")
    return modules


def ensure_padding_token(tokenizer) -> None:
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
        return
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)


def filter_generate_inputs(model, enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "token_type_ids" not in enc:
        return enc
    try:
        forward_params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        forward_params = {}
    if "token_type_ids" in forward_params:
        return enc
    enc = dict(enc)
    enc.pop("token_type_ids", None)
    return enc


@torch.no_grad()
def generate_one(tokenizer, model, prompt: str, max_new_tokens: int, do_sample: bool,
                 temperature: float, top_p: float, num_return_sequences: int = 1):
    ensure_padding_token(tokenizer)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}
    enc = filter_generate_inputs(model, enc)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})
    out = model.generate(**enc, **gen_kwargs)
    prompt_len = enc["input_ids"].shape[1]
    texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return out, prompt_len, texts


@torch.no_grad()
def batched_generate_texts(tokenizer, model, prompts: list[str], max_new_tokens: int,
                           batch_size: int) -> list[str]:
    results: list[str] = []
    ensure_padding_token(tokenizer)
    pad_side_orig = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            dev = next(model.parameters()).device
            enc = {k: v.to(dev) for k, v in enc.items()}
            enc = filter_generate_inputs(model, enc)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = enc["input_ids"].shape[1]
            results.extend(tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True))
    finally:
        tokenizer.padding_side = pad_side_orig
    return results


def token_stats(model, sequences: torch.Tensor, prompt_len: int, pad_id: int):
    logits = model(sequences).logits
    logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    targets = sequences[:, 1:]
    gathered = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
    response_mask = positions >= max(0, prompt_len - 1)
    if pad_id is not None:
        response_mask = response_mask & (targets != pad_id)
    return gathered, response_mask, entropy


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    if not mask.any():
        return None
    return float(x.masked_select(mask).mean().detach().cpu())


def masked_std(x: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    if not mask.any():
        return None
    vals = x.masked_select(mask)
    if vals.numel() < 2:
        return 0.0
    return float(vals.std(unbiased=False).detach().cpu())


def ratio_diagnostics(
    ratio: torch.Tensor,
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    old_entropy: torch.Tensor,
    new_entropy: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
) -> Dict[str, Optional[float]]:
    if not mask.any():
        return {
            "ratio_mean": None,
            "ratio_std": None,
            "ratio_min": None,
            "ratio_max": None,
            "clip_fraction": None,
            "clip_high_fraction": None,
            "clip_low_fraction": None,
            "approx_kl": None,
            "old_entropy_mean": None,
            "new_entropy_mean": None,
            "entropy_delta_mean": None,
        }
    vals = ratio.masked_select(mask)
    high = (ratio > 1.0 + clip_eps) & mask
    low = (ratio < 1.0 - clip_eps) & mask
    denom = float(mask.sum().detach().cpu())
    old_ent = old_entropy.masked_select(mask)
    new_ent = new_entropy.masked_select(mask)
    return {
        "ratio_mean": float(vals.mean().detach().cpu()),
        "ratio_std": masked_std(ratio, mask),
        "ratio_min": float(vals.min().detach().cpu()),
        "ratio_max": float(vals.max().detach().cpu()),
        "clip_fraction": float((high | low).sum().detach().cpu()) / denom,
        "clip_high_fraction": float(high.sum().detach().cpu()) / denom,
        "clip_low_fraction": float(low.sum().detach().cpu()) / denom,
        "approx_kl": masked_mean(old_logp - new_logp, mask),
        "old_entropy_mean": float(old_ent.mean().detach().cpu()),
        "new_entropy_mean": float(new_ent.mean().detach().cpu()),
        "entropy_delta_mean": float((new_ent - old_ent).mean().detach().cpu()),
    }


def compute_rewards(texts: list[str], ex: Dict[str, str], reward_mode: str,
                    random_reward_p: float, rng: random.Random) -> list[float]:
    if reward_mode == "random":
        return [1.0 if rng.random() < random_reward_p else 0.0 for _ in texts]
    if reward_mode == "random_label":
        idx = rng.randint(0, len(texts) - 1)
        return [1.0 if i == idx else 0.0 for i in range(len(texts))]
    if reward_mode == "format":
        return [1.0 if "####" in text or "\\boxed" in text else 0.0 for text in texts]
    if reward_mode == "zero":
        return [0.0 for _ in texts]

    if reward_mode == "noisy_gt":
        gold = gold_number(ex)
        rewards = []
        for text in texts:
            got = extract_final_number(text)
            correct = gold is not None and got is not None and math.isclose(got, gold, rel_tol=0.0, abs_tol=1e-6)
            r = 1.0 if correct else 0.0
            if rng.random() < random_reward_p:
                r = 1.0 - r  # flip
            rewards.append(r)
        return rewards

    gold = gold_number(ex)
    rewards = []
    for text in texts:
        got = extract_final_number(text)
        correct = gold is not None and got is not None and math.isclose(got, gold, rel_tol=0.0, abs_tol=1e-6)
        if reward_mode == "ground_truth":
            rewards.append(1.0 if correct else 0.0)
        elif reward_mode == "anti_gt":
            rewards.append(0.0 if correct else 1.0)
        elif reward_mode == "incorrect_any":
            rewards.append(1.0 if got is not None and not correct else 0.0)
        else:
            raise ValueError(f"Unknown reward mode: {reward_mode}")
    return rewards


def pearson_corr(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def rollout_diagnostics(texts: list[str], rewards: list[float], ex: Dict[str, str]) -> Dict[str, Optional[float]]:
    gold = gold_number(ex)
    reward_vals = [float(r) for r in rewards]
    lenient_correct = []
    strict_correct = []
    extracted = []
    strict_extracted = []
    format_ok = []
    token_counts = []
    for text in texts:
        got = extract_final_number(text)
        strict_got = extract_strict_hash_number(text)
        lenient_correct.append(float(gold is not None and got is not None and math.isclose(got, gold, rel_tol=0.0, abs_tol=1e-6)))
        strict_correct.append(float(gold is not None and strict_got is not None and math.isclose(strict_got, gold, rel_tol=0.0, abs_tol=1e-6)))
        extracted.append(float(got is not None))
        strict_extracted.append(float(strict_got is not None))
        format_ok.append(float("####" in text or "\\boxed" in text))
        token_counts.append(float(len(re.findall(r"\S+", text))))
    return {
        "rollout_lenient_correct_rate": sum(lenient_correct) / max(1, len(lenient_correct)),
        "rollout_strict_correct_rate": sum(strict_correct) / max(1, len(strict_correct)),
        "rollout_extract_rate": sum(extracted) / max(1, len(extracted)),
        "rollout_strict_extract_rate": sum(strict_extracted) / max(1, len(strict_extracted)),
        "rollout_format_rate": sum(format_ok) / max(1, len(format_ok)),
        "rollout_response_tokens_mean": sum(token_counts) / max(1, len(token_counts)),
        "reward_corr_lenient_correct": pearson_corr(reward_vals, lenient_correct),
        "reward_corr_strict_correct": pearson_corr(reward_vals, strict_correct),
        "reward_corr_extract": pearson_corr(reward_vals, extracted),
        "reward_corr_strict_extract": pearson_corr(reward_vals, strict_extracted),
        "reward_corr_format": pearson_corr(reward_vals, format_ok),
        "reward_corr_response_tokens": pearson_corr(reward_vals, token_counts),
        "rollout_rewards": reward_vals,
        "rollout_correct": lenient_correct,
        "rollout_strict_correct": strict_correct,
    }


@torch.no_grad()
def evaluate(tokenizer, model, rows: list[Dict[str, str]], mode: str,
             max_new_tokens: int, limit: int, batch_size: int,
             task_type: str = "math") -> Dict[str, Any]:
    model.eval()
    subset = rows[:limit] if limit > 0 else rows
    correct = 0
    strict_correct = 0
    extracted = 0
    strict_extracted = 0
    format_ok = 0
    code_like = 0
    feature_sums: Dict[str, float] = {}
    examples = []
    outcomes = []
    if task_type == "code":
        prompts = [build_code_prompt(ex) for ex in subset]
    else:
        prompts = [build_prompt(ex, mode) for ex in subset]
    texts = batched_generate_texts(tokenizer, model, prompts, max_new_tokens, batch_size)
    if task_type == "code" and HAS_CODE_EVAL:
        from code_eval import evaluate_code as _eval_code
        code_results = _eval_code(texts, subset)
        model.train()
        return code_results
    for idx, (ex, text) in enumerate(zip(subset, texts)):
        got = extract_final_number(text)
        strict_got = extract_strict_hash_number(text)
        gold = gold_number(ex)
        ok = gold is not None and got is not None and math.isclose(got, gold, rel_tol=0.0, abs_tol=1e-6)
        strict_ok = gold is not None and strict_got is not None and math.isclose(strict_got, gold, rel_tol=0.0, abs_tol=1e-6)
        correct += int(ok)
        strict_correct += int(strict_ok)
        extracted += int(got is not None)
        strict_extracted += int(strict_got is not None)
        format_ok += int("####" in text or "\\boxed" in text)
        feats = behavior_features(text)
        for key, val in feats.items():
            feature_sums[key] = feature_sums.get(key, 0.0) + val
        code_like += int(feats["code_like"])
        outcomes.append({
            "idx": idx,
            "gold": gold,
            "got": got,
            "strict_got": strict_got,
            "correct": bool(ok),
            "strict_correct": bool(strict_ok),
            "extract_ok": got is not None,
            "strict_extract_ok": strict_got is not None,
            "format_ok": bool("####" in text or "\\boxed" in text),
            "response_tokens": feats["response_tokens"],
            "code_like": bool(feats["code_like"]),
        })
        if len(examples) < 3:
            examples.append({"question": ex["orig_question"], "gold": gold, "got": got, "pred": text[:500]})
    denom = max(1, len(subset))
    model.train()
    result = {
        "accuracy": correct / denom,
        "strict_accuracy": strict_correct / denom,
        "extract_rate": extracted / denom,
        "strict_extract_rate": strict_extracted / denom,
        "format_rate": format_ok / denom,
        "code_like_rate": code_like / denom,
        "n": len(subset),
        "examples": examples,
        "outcomes": outcomes,
    }
    for key, val in feature_sums.items():
        result[f"{key}_mean"] = val / denom
    return result


def append_jsonl(path: str, rec: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--base_model", type=str, default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--train_mode", choices=["original", "processed_m", "processed_jm"], default="original")
    ap.add_argument("--reward_mode", choices=["ground_truth", "random", "random_label", "format", "zero", "incorrect_any", "noisy_gt", "anti_gt"], default="random")
    ap.add_argument("--steer_steps", type=int, default=0,
                    help="For first K steps use --steer_reward_mode instead of --reward_mode (critical-window steering)")
    ap.add_argument("--steer_reward_mode", choices=["ground_truth", "anti_gt"], default="ground_truth",
                    help="Reward mode during the steering window")
    ap.add_argument("--entropy_coeff", type=float, default=0.0,
                    help="Entropy bonus coefficient added to loss (encourages exploration); 0 = disabled")
    ap.add_argument("--random_reward_p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--train_percent", type=float, default=1.0)
    ap.add_argument("--eval_limit", type=int, default=100)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--train_steps", type=int, default=50)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--kl_coeff", type=float, default=0.0, help="KL penalty vs frozen reference model; 0 = disabled")
    ap.add_argument("--ppo_epochs", type=int, default=1)
    ap.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps (simulate larger batch)")
    ap.add_argument("--grad_checkpointing", action="store_true", help="Enable gradient checkpointing to reduce activation memory (needed for large group_size on code task)")
    ap.add_argument("--disable_clipping", action="store_true")
    ap.add_argument("--no_update", action="store_true")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_target_modules", type=str, default="auto")
    ap.add_argument("--run_suffix", type=str, default="")
    ap.add_argument("--pretrain_sft_data", type=str, default=None,
                    help="Path to JSONL for pre-intervention SFT before RLVR (D3 format vs reasoning)")
    ap.add_argument("--pretrain_sft_steps", type=int, default=200,
                    help="Number of SFT gradient steps for pre-intervention")
    ap.add_argument("--pretrain_sft_lr", type=float, default=2e-5,
                    help="Learning rate for pre-intervention SFT")
    ap.add_argument("--task_type", choices=["math", "code"], default="math",
                        help="math = GSM number extraction; code = MBPP code execution")
    ap.add_argument("--save_adapter", action="store_true",
                    help="Save LoRA adapter after training")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    revision_name = args.revision or "main"
    model_name = args.base_model.split("/")[-1]
    run_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.run_suffix.strip()).strip("-")
    suffix_part = f"_{run_suffix}" if run_suffix else ""
    run_name = f"rlvr_{model_name}_{revision_name}_{args.train_mode}_{args.reward_mode}_seed{args.seed}{suffix_part}"
    metrics_path = os.path.join(args.output_dir, f"{run_name}.jsonl")

    rows = load_jsonl(args.dataset_path)
    train_rows, eval_rows = split_train_eval(rows, args.train_frac, args.seed)
    k = max(1, int(len(train_rows) * args.train_percent))
    train_rows = train_rows[:k]

    try:
        tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision, use_fast=True)
    except Exception:
        if not args.revision:
            raise
        tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    ensure_padding_token(tok)

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_for(args.base_model, args.lora_target_modules),
    )
    model = get_peft_model(model, lora_cfg)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()

    if args.pretrain_sft_data:
        sft_rows = load_jsonl(args.pretrain_sft_data)
        sft_opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.pretrain_sft_lr,
        )
        sft_rng = random.Random(args.seed + 31)
        print(f"[pre-intervention SFT] {len(sft_rows)} rows, {args.pretrain_sft_steps} steps", flush=True)
        for sft_step in range(1, args.pretrain_sft_steps + 1):
            ex = sft_rng.choice(sft_rows)
            if args.task_type == "code":
                text = build_code_prompt(ex) + " " + ex.get("orig_answer", "")
            else:
                q, a, instr = fields_for_mode(ex, args.train_mode)
                text = f"{instr}\n\nQuestion: {q}\nAnswer: {a}"
            enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            out = model(**enc, labels=labels)
            sft_opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sft_opt.step()
            if sft_step % 50 == 0:
                print(f"  [sft step {sft_step}] loss={out.loss.item():.4f}", flush=True)
        del sft_opt
        print(f"[pre-intervention SFT] done", flush=True)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    opt.zero_grad(set_to_none=True)  # init for gradient accumulation
    rng = random.Random(args.seed + 17)

    meta = {
        "phase": "meta",
        "time": time.time(),
        "base_model": args.base_model,
        "revision": revision_name,
        "train_mode": args.train_mode,
        "reward_mode": args.reward_mode,
        "random_reward_p": args.random_reward_p,
        "seed": args.seed,
        "train_examples": len(train_rows),
        "eval_examples": min(args.eval_limit, len(eval_rows)),
        "train_steps": args.train_steps,
        "group_size": args.group_size,
        "clip_eps": args.clip_eps,
        "ppo_epochs": args.ppo_epochs,
        "accum_steps": args.accum_steps,
        "disable_clipping": args.disable_clipping,
        "no_update": args.no_update,
        "lora_dropout": args.lora_dropout,
        "run_suffix": run_suffix,
        "pretrain_sft_data": args.pretrain_sft_data,
        "pretrain_sft_steps": args.pretrain_sft_steps if args.pretrain_sft_data else 0,
        "task_type": args.task_type,
        "steer_steps": args.steer_steps,
        "steer_reward_mode": args.steer_reward_mode if args.steer_steps > 0 else None,
        "entropy_coeff": args.entropy_coeff,
    }
    append_jsonl(metrics_path, meta)

    before = evaluate(tok, model, eval_rows, args.train_mode, args.max_new_tokens, args.eval_limit, args.eval_batch_size, task_type=args.task_type)
    append_jsonl(metrics_path, {"phase": "before", "step": 0, "time": time.time(), **before})
    print(f"[before] acc={before['accuracy']:.3f} format={before['format_rate']:.3f} extract={before['extract_rate']:.3f}", flush=True)

    pad_id = tok.pad_token_id
    last_eval_step: Optional[int] = None
    last_eval: Optional[Dict[str, Any]] = None
    for step in range(1, args.train_steps + 1):
        ex = train_rows[(step - 1) % len(train_rows)]
        prompt = build_code_prompt(ex) if args.task_type == "code" else build_prompt(ex, args.train_mode)
        model.eval()
        with torch.no_grad():
            sequences, prompt_len, texts = generate_one(
                tok, model, prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.group_size,
            )
            old_logp, mask, old_entropy = token_stats(model, sequences, prompt_len, pad_id)
            # Reference logprobs (frozen base model w/o LoRA) for KL penalty
            if args.kl_coeff > 0:
                with model.disable_adapter():
                    ref_logp, _, _ = token_stats(model, sequences, prompt_len, pad_id)
            _reward_fn = compute_code_rewards if args.task_type == "code" else compute_rewards
            effective_reward_mode = args.reward_mode
            if args.steer_steps > 0 and step <= args.steer_steps:
                effective_reward_mode = args.steer_reward_mode
            rewards = torch.tensor(
                _reward_fn(texts, ex, effective_reward_mode, args.random_reward_p, rng),
                dtype=torch.float32,
                device=sequences.device,
            )
            reward_list = [float(x) for x in rewards.detach().cpu().tolist()]
            train_diag = rollout_diagnostics(texts, reward_list, ex)
            if rewards.numel() > 1 and rewards.std(unbiased=False) > 1e-6:
                adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            else:
                adv = rewards - rewards.mean()

        last_diag: Dict[str, Optional[float]] = {}
        last_loss: Optional[torch.Tensor] = None
        epochs = max(1, args.ppo_epochs)
        for ppo_epoch in range(1, epochs + 1):
            model.train()
            new_logp, new_mask, new_entropy = token_stats(model, sequences, prompt_len, pad_id)
            train_mask = mask & new_mask
            ratio = torch.exp((new_logp - old_logp.detach()).clamp(-10, 10))
            adv_tok = adv.unsqueeze(1).expand_as(ratio)
            unclipped = ratio * adv_tok
            clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_tok
            objective = unclipped if args.disable_clipping else torch.minimum(unclipped, clipped)
            if train_mask.any():
                loss = -objective.masked_select(train_mask).mean()
                if args.kl_coeff > 0:
                    kl = (new_logp - ref_logp.detach()).masked_select(train_mask).mean()
                    loss = loss + args.kl_coeff * kl
                if args.entropy_coeff > 0:
                    ent_bonus = new_entropy.masked_select(train_mask).mean()
                    loss = loss - args.entropy_coeff * ent_bonus
            else:
                loss = objective.sum() * 0.0
            last_loss = loss
            last_diag = ratio_diagnostics(
                ratio,
                old_logp.detach(),
                new_logp,
                old_entropy.detach(),
                new_entropy,
                train_mask,
                args.clip_eps,
            )
            if not args.no_update:
                (loss / args.accum_steps).backward()
                if step % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

        rec = {
            "phase": "train",
            "step": step,
            "time": time.time(),
            "effective_reward_mode": effective_reward_mode,
            "loss": float(last_loss.detach().cpu()) if last_loss is not None else None,
            "ppo_epochs": epochs,
            "reward_mean": float(rewards.mean().detach().cpu()),
            "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
            "nonzero_adv": int((adv.abs() > 1e-6).sum().detach().cpu()),
            **last_diag,
            **train_diag,
        }
        append_jsonl(metrics_path, rec)
        if step % args.eval_every == 0 or step == args.train_steps:
            ev = evaluate(tok, model, eval_rows, args.train_mode, args.max_new_tokens, args.eval_limit, args.eval_batch_size, task_type=args.task_type)
            append_jsonl(metrics_path, {"phase": "eval", "step": step, "time": time.time(), **ev})
            last_eval_step = step
            last_eval = ev
            print(f"[step {step}] loss={rec['loss']:.4f} reward={rec['reward_mean']:.3f} acc={ev['accuracy']:.3f}", flush=True)

    if last_eval_step == args.train_steps and last_eval is not None:
        after = last_eval
    else:
        after = evaluate(tok, model, eval_rows, args.train_mode, args.max_new_tokens, args.eval_limit, args.eval_batch_size, task_type=args.task_type)
    append_jsonl(metrics_path, {"phase": "after", "step": args.train_steps, "time": time.time(), **after})
    print(f"[after] acc={after['accuracy']:.3f} metrics={metrics_path}", flush=True)

    if args.save_adapter:
        adapter_dir = os.path.join(args.output_dir, f"adapter_{run_suffix}")
        model.save_pretrained(adapter_dir)
        print(f"[saved adapter] {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
