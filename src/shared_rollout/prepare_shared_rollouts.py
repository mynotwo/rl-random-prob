#!/usr/bin/env python3
"""Generate ONE frozen-source rollout stream: 500 groups x 4 sampled answers, plus the replicate's shared rewards/advantages.

    prepare_shared_rollouts.py --root ROOT --source S1 --replicate 51001 [--groups 500]

The source is the raw checkpoint (no adapter, eval(), no_grad). Raw token ids are cached; learners never re-tokenize.
Nothing here looks at the gold answer: no retry, no filtering, no balancing. Needs a GPU.
"""
import argparse, json, pathlib, random, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sr_common as C

PROMPT_BATCH = 10      # prompts per generate call (x4 samples). Fixed: it is part of the RNG layout of the streams.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--source", required=True, choices=list(C.SOURCES))
    ap.add_argument("--replicate", type=int, required=True); ap.add_argument("--groups", type=int, default=C.STEPS)
    a = ap.parse_args()
    out = pathlib.Path(a.root) / "rollouts" / a.source / f"replicate{a.replicate}"
    if (out / "meta.json").exists():
        print(f"skip: {out} already complete"); return
    out.mkdir(parents=True, exist_ok=True)
    branch, sha = C.CKPT[C.SOURCES[a.source]]
    train_rows, _, train_ids, _ = C.load_split()
    rewards = C.reward_table(a.replicate)
    rt = pathlib.Path(a.root) / "reward_tables" / f"replicate{a.replicate}.json"; rt.parent.mkdir(parents=True, exist_ok=True)
    table = dict(replicate=a.replicate, seed=C.SEED_BASE["reward"] + a.replicate, rewards=rewards.int().tolist())
    if rt.exists():
        assert json.load(open(rt)) == table, "reward table on disk differs"       # both sources must see the same table
    else:
        C.write_atomic(rt, json.dumps(table))

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=sha, use_fast=True); C.ensure_padding_token(tok); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, revision=sha, dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda().eval()
    t_load = time.time() - t0
    seed = C.SEED_BASE[a.source] + a.replicate
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gen_cfg = dict(max_new_tokens=C.MAX_NEW, do_sample=True, temperature=1.0, top_p=0.95, num_return_sequences=C.G,
                   eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    groups, t1 = [], time.time()
    with torch.no_grad():
        for s in range(0, a.groups, PROMPT_BATCH):
            rows = [train_rows[t % len(train_rows)] for t in range(s, min(s + PROMPT_BATCH, a.groups))]   # step t+1 reads train_rows[t % 800]
            prompts = [C.build_prompt(ex, "original") for ex in rows]
            enc = tok(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            enc = C.filter_generate_inputs(model, {k: v.cuda() for k, v in enc.items()})
            new = model.generate(**enc, **gen_cfg)[:, enc["input_ids"].shape[1]:].tolist()
            for i, (ex, prompt) in enumerate(zip(rows, prompts)):
                resp = []
                for ids in new[i * C.G:(i + 1) * C.G]:
                    if tok.eos_token_id in ids:
                        ids = ids[:ids.index(tok.eos_token_id) + 1]            # keep the EOS it produced, drop the padding after it
                    resp.append(ids)
                step = s + i + 1; r = rewards[step - 1]
                g = dict(source_id=a.source, source_sha=sha, replicate_id=a.replicate, step=step, train_item_id=train_ids[(step - 1) % len(train_ids)],
                         prompt_text=prompt, prompt_input_ids=tok(prompt, truncation=True, max_length=1024)["input_ids"],
                         response_input_ids=resp, response_text=tok.batch_decode(resp, skip_special_tokens=True), generated_lengths=[len(x) for x in resp],
                         response_loss_masks=[[int(t != tok.pad_token_id) for t in x] for x in resp],
                         rewards=r.int().tolist(), advantages=C.advantages(r).tolist())
                g["prompt_length"] = len(g["prompt_input_ids"]); g["group_hash"] = C.group_hash(g)
                groups.append(g)
            print(f"[{a.source} b={a.replicate}] {len(groups)}/{a.groups} groups, {time.time() - t1:.0f}s", flush=True)
    C.write_atomic(out / "groups.jsonl", "".join(json.dumps(g) + "\n" for g in groups))
    snap = pathlib.Path(tok.name_or_path)
    meta = dict(protocol_id=C.PROTOCOL_ID, source_id=a.source, model=C.MODEL, branch=branch, source_sha=sha, replicate_id=a.replicate, groups=len(groups),
                rollout_seed=seed, reward_seed=C.SEED_BASE["reward"] + a.replicate, prompt_batch=PROMPT_BATCH, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                mask_convention="response tokens that are not pad; pad != eos for OLMo 2, so a generated EOS is inside the loss mask",
                generation_config={k: v for k, v in gen_cfg.items()}, tokenizer_vocab_hash=C.sha_obj(tok.get_vocab()), dataset_sha256=C.DATASET_SHA,
                content_hash=C.sha_file(out / "groups.jsonl"), seconds_load=round(t_load, 1), seconds_generate=round(time.time() - t1, 1),
                gpu=torch.cuda.get_device_name(0), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), torch=torch.__version__)
    C.write_atomic(out / "meta.json", json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("source_id", "replicate_id", "groups", "seconds_generate", "content_hash")}))


if __name__ == "__main__":
    main()
