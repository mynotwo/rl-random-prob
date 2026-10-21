#!/usr/bin/env python3
"""Train ONE learner on ONE cached source stream (off-policy random-advantage adaptation).

    train_shared_rollouts.py --root ROOT --source S1 --learner L2 --replicate 51001 [--steps 500 --eval_every 100]

The learner never generates its own training answers and never sees gold answers: the update reads only cached token ids,
masks and advantages. No ratio / importance weight / PPO clipping / KL / entropy / SFT term exists in the loss.
The training path draws no random numbers (cached data, fixed adapter start, dropout 0, greedy eval), so a failed
run is simply rerun from step 0 under the same job key with the attempt counter raised. Needs a GPU.
"""
import argparse, json, math, pathlib, sys, time

import torch

import sr_common as C

LR, GRAD_CAP = 3e-5, 1.0
USED_KEYS = {"prompt_input_ids", "response_input_ids", "response_loss_masks", "advantages", "rewards", "step", "group_hash"}   # all a learner may read


def attach_lora(model, b):
    """LoRA r8/a16/dropout0 on the scan's seven projections, started from replicate b's shared tensors. Base stays frozen."""
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                             target_modules=C.target_modules_for(C.MODEL, "auto")))
    lora = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert lora and all("lora_" in n for n in lora), "something other than LoRA is trainable"
    init = C.lora_init_tensors({n: p.shape for n, p in lora.items()}, b)
    with torch.no_grad():
        for n, p in lora.items():
            p.copy_(init[n].to(p.device, p.dtype))
    return model, lora, C.tensors_hash(init)


def make_optimizer(lora):
    return torch.optim.AdamW(lora.values(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def update(model, opt, lora, seq, mask, adv):
    """One scheduled step: always backward + norm cap + optimizer.step, also when every advantage is zero."""
    model.train()
    loss, logp = C.offpolicy_loss(model(seq).logits, seq, mask, adv)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(list(lora.values()), GRAD_CAP))     # returns the norm BEFORE capping
    lv = loss.item(); finite = math.isfinite(lv) and math.isfinite(gn)
    if finite:
        opt.step()
    n = int(mask.sum())
    return dict(loss=lv, grad_norm=gn, grad_capped=gn > GRAD_CAP, valid_tokens=n, finite=finite,
                mean_logp=float(logp.masked_select(mask).mean()) if n else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--source", required=True, choices=list(C.SOURCES))
    ap.add_argument("--learner", required=True, choices=list(C.LEARNERS)); ap.add_argument("--replicate", type=int, required=True)
    ap.add_argument("--steps", type=int, default=C.STEPS); ap.add_argument("--eval_every", type=int, default=100)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = pathlib.Path(a.root); out = root / "runs" / f"{a.source}_{a.learner}" / f"replicate{a.replicate}"
    if (out / "summary.json").exists():
        print(f"skip: {out} already complete"); return
    out.mkdir(parents=True, exist_ok=True)
    attempt = len(list(out.glob("attempt*.json"))) + 1
    for f in list(out.glob("eval_step*.jsonl")) + [out / "train_metrics.jsonl"]:
        f.unlink(missing_ok=True)                                                  # a rerun starts from step 0
    branch, sha = C.CKPT[C.LEARNERS[a.learner]]
    groups, cmeta = C.read_cache(root / "rollouts" / a.source / f"replicate{a.replicate}", a.steps)
    assert all(C.group_hash(g) == g["group_hash"] for g in groups), "a cached group does not match its hash"
    job_key = dict(source_sha=cmeta["source_sha"], learner_sha=sha, replicate_id=a.replicate, protocol_id=C.PROTOCOL_ID, steps=a.steps)
    C.write_atomic(out / f"attempt{attempt}.json", json.dumps(dict(job_key=job_key, started=time.time())))

    t0 = time.time()
    _, eval_rows, _, eval_ids = C.load_split()
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=sha, use_fast=True); C.ensure_padding_token(tok)
    assert C.sha_obj(tok.get_vocab()) == cmeta["tokenizer_vocab_hash"] and tok.pad_token_id == cmeta["pad_token_id"], "tokenizer differs from the source's"
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, revision=sha, dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
    model.config.use_cache = False
    probe = [p for n, p in model.named_parameters()][::97]                         # frozen-base check on a fixed sample of tensors
    base_before = [float(p.double().sum()) for p in probe]
    model, lora, init_hash = attach_lora(model, a.replicate)
    ih = root / "adapter_initializations" / f"replicate{a.replicate}.json"; ih.parent.mkdir(parents=True, exist_ok=True)
    if ih.exists():
        assert json.load(open(ih))["tensors_hash"] == init_hash, "adapter start differs from the other cells of this replicate"
    else:
        C.write_atomic(ih, json.dumps(dict(replicate=a.replicate, seed=C.SEED_BASE["lora"] + a.replicate, tensors_hash=init_hash,
                                           n_tensors=len(lora), trainable_parameters=sum(p.numel() for p in lora.values()))))
    opt = make_optimizer(lora)
    t_load = time.time() - t0
    config = dict(job_key=job_key, attempt=attempt, source=a.source, learner=a.learner, learner_branch=branch, cache_content_hash=cmeta["content_hash"],
                  adapter_init_hash=init_hash, trainable_parameters=sum(p.numel() for p in lora.values()), lora_targets=C.target_modules_for(C.MODEL, "auto"),
                  lr=LR, schedule="constant", optimizer="AdamW betas=(0.9,0.999) eps=1e-8 weight_decay=0", grad_norm_cap=GRAD_CAP, kl_coeff=0, entropy_coeff=0,
                  loss="-(A*logp) token-mean over the group's cached response mask; FP32 log_softmax; no ratio, no clipping, no KL, no entropy",
                  model_dtype="bfloat16", lora_dtype=str(next(iter(lora.values())).dtype), gradient_checkpointing=False, eval="greedy, 96 new tokens, batch 8, 200 items",
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    C.write_atomic(out / "config.json", json.dumps(config, indent=1))

    metrics = open(out / "train_metrics.jsonl", "w"); evals, t_eval, t_train = {}, 0.0, 0.0

    def run_eval(step):
        nonlocal t_eval
        t = time.time(); recs, summ = C.greedy_eval(tok, model, eval_rows, eval_ids); t_eval += time.time() - t
        C.write_atomic(out / f"eval_step{step}.jsonl", "".join(json.dumps(r) + "\n" for r in recs))
        evals[step] = summ; metrics.write(json.dumps(dict(phase="eval", step=step, **summ)) + "\n"); metrics.flush()
        print(f"[{a.source}->{a.learner} b={a.replicate}] step {step} acc {summ['accuracy']:.3f} strict {summ['strict_accuracy']:.3f} fmt {summ['format_rate']:.2f}", flush=True)

    run_eval(0)
    prev = [p.detach().clone() for p in lora.values()]
    for step in range(1, a.steps + 1):
        g = {k: groups[step - 1][k] for k in USED_KEYS}                            # gold answers and correctness are not even in scope
        t = time.time()
        seq, mask, adv = C.group_tensors(g, tok.pad_token_id, "cuda")
        rec = update(model, opt, lora, seq, mask, adv)
        with torch.no_grad():
            upd = math.sqrt(sum(float(((p - q) ** 2).sum()) for p, q in zip(lora.values(), prev)))
            for p, q in zip(lora.values(), prev):
                q.copy_(p)
        t_train += time.time() - t
        r = torch.tensor(g["rewards"], dtype=torch.float32)
        metrics.write(json.dumps(dict(phase="train", step=step, cache_key=g["group_hash"], reward_mean=float(r.mean()), adv_abs_max=float(adv.abs().max()),
                                      nonzero_adv=int((adv.abs() > 1e-6).sum()), update_norm=upd, time=time.time(), **rec)) + "\n")
        if not rec["finite"]:
            metrics.close(); model.save_pretrained(out / f"adapter_failed_step{step}")
            C.write_atomic(out / "failure.json", json.dumps(dict(kind="numerical", step=step, **rec)))
            sys.exit(3)
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
    metrics.close()
    model.save_pretrained(out / f"adapter_step{a.steps}")
    base_after = [float(p.double().sum()) for p in probe]
    acc = {s: e["accuracy"] for s, e in evals.items()}; peak = max((s for s in acc if s > 0), key=acc.get)
    C.write_atomic(out / "timing.json", json.dumps(dict(seconds_load=round(t_load, 1), seconds_train=round(t_train, 1), seconds_eval=round(t_eval, 1),
                                                         seconds_total=round(time.time() - t0, 1), sec_per_step=round(t_train / a.steps, 3),
                                                         peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), gpu=torch.cuda.get_device_name(0))))
    C.write_atomic(out / "summary.json", json.dumps(dict(job_key=job_key, attempt=attempt, source=a.source, learner=a.learner, replicate=a.replicate,
                                                          step0_accuracy=acc[0], terminal_accuracy=acc[a.steps], gain=round(acc[a.steps] - acc[0], 6),
                                                          strict_gain=round(evals[a.steps]["strict_accuracy"] - evals[0]["strict_accuracy"], 6),
                                                          peak_accuracy=acc[peak], peak_step=peak, evals=evals, base_frozen=base_before == base_after,
                                                          adapter_init_hash=init_hash, adapter_final_hash=C.tensors_hash(lora),
                                                          cache_content_hash=cmeta["content_hash"]), indent=1))
    print(f"done: step0 {acc[0]:.3f} -> {acc[a.steps]:.3f}, base_frozen={base_before == base_after}")


if __name__ == "__main__":
    main()
