"""P1 / P2 shared-rollout 2x2 cross-training experiment: everything the four scripts share.

Frozen sources generate the rollouts once; two learners replay the SAME token ids, rewards and advantages.
The learner loss is -(A * log pi(y)) averaged over the group's response tokens: no ratio, no importance weight,
no PPO clipping, no KL, no entropy term. Data, split, prompt, LoRA targets and scorers come from the audited
September trainer (imported, not copied).
"""
import hashlib, json, math, os, pathlib, sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
MS = HERE.parent
sys.path[:0] = [str(MS), str(MS / "trainer")]
from probe_checkpoints import extract_final_number, extract_strict_hash_number, load_jsonl  # noqa: E402
from protocol import split_rows  # noqa: E402
from rlvr_probe_v2_fixedsplit import build_prompt, ensure_padding_token, filter_generate_inputs, target_modules_for  # noqa: E402

MODEL = "allenai/OLMo-2-1124-7B"
CKPT = {"P1": ("stage1-step928646-tokens3896B", "c0371f4281bf2376207646c6b62ddc6c442c7577"),
        "P2": ("stage2-ingredient1-step1000-tokens5B", "ae389638809610ab9d1f75b2e53b120b324c5c43")}
SOURCES, LEARNERS = {"S1": "P1", "S2": "P2"}, {"L1": "P1", "L2": "P2"}
DATASET = MS / "data" / "synthetic_gsm_rlvr_1000_seed13.jsonl"
DATASET_SHA = "7891bee44787119a901ce60a04a94f8c4a36b637ab22fd48106bf3f66b553a72"
REPLICATES, SMOKE_REPLICATE = list(range(51001, 51017)), 59001
G, STEPS, MAX_NEW, EVAL_BATCH = 4, 500, 96, 8
SEED_BASE = {"S1": 100000, "S2": 200000, "reward": 300000, "lora": 400000}
PROTOCOL_ID = "shared_rollout_offpolicy_v1"
PY = os.environ.get("PROBE_PYTHON", "python3")


def sha_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def write_atomic(path, text):
    tmp = pathlib.Path(f"{path}.tmp{os.getpid()}"); tmp.write_text(text); os.replace(tmp, path)


def load_split():
    """The scan's split 777: 800 train rows in their fixed order, 200 eval rows, and their row ids."""
    assert sha_file(DATASET) == DATASET_SHA, "dataset file differs from the September scan"
    return split_rows(load_jsonl(str(DATASET)), 0.8, 777)


def reward_table(b):
    """500x4 Bernoulli(0.5) table of replicate b from its own CPU generator. Always drawn at full size, then sliced."""
    g = torch.Generator(device="cpu").manual_seed(SEED_BASE["reward"] + b)
    return torch.bernoulli(torch.full((STEPS, G), 0.5), generator=g)


def advantages(r):
    """Group advantage exactly as the reference trainer computes it (population sd, eps 1e-6)."""
    r = r.float(); sd = r.std(unbiased=False)
    return (r - r.mean()) / (sd + 1e-6) if sd > 1e-6 else r - r.mean()


def lora_init_tensors(shapes, b):
    """Shared LoRA start of replicate b, drawn on CPU so it never depends on the device or on model loading.
    A ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (= PEFT's kaiming_uniform(a=sqrt(5))), B = 0. `shapes`: {param name: shape}."""
    g = torch.Generator(device="cpu").manual_seed(SEED_BASE["lora"] + b)
    out = {}
    for name in sorted(shapes):
        shape = tuple(shapes[name])
        if "lora_A" in name:
            bound = 1.0 / math.sqrt(shape[1])
            out[name] = (torch.rand(shape, generator=g, dtype=torch.float32) * 2 - 1) * bound
        else:
            out[name] = torch.zeros(shape, dtype=torch.float32)
    return out


def tensors_hash(named):
    h = hashlib.sha256()
    for name in sorted(named):
        h.update(name.encode()); h.update(named[name].detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def group_hash(g):
    return sha_obj({k: g[k] for k in ("step", "train_item_id", "prompt_input_ids", "response_input_ids", "response_loss_masks", "rewards", "advantages")})


def group_tensors(g, pad_id, device):
    """[4, L] right-padded sequences, the [4, L-1] mask over next-token targets, and the cached advantages.
    Mask = response positions whose cached loss mask is 1 (the reference convention: response tokens that are not pad;
    OLMo 2 has pad != eos, so a generated EOS is inside the mask)."""
    p, resp, lm = g["prompt_input_ids"], g["response_input_ids"], g["response_loss_masks"]
    L = len(p) + max(len(r) for r in resp)
    seq = torch.full((len(resp), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(resp), L - 1), dtype=torch.bool)
    for i, (r, m) in enumerate(zip(resp, lm)):
        seq[i, :len(p)] = torch.tensor(p); seq[i, len(p):len(p) + len(r)] = torch.tensor(r)
        mask[i, len(p) - 1:len(p) - 1 + len(r)] = torch.tensor(m, dtype=torch.bool)
    return seq.to(device), mask.to(device), torch.tensor(g["advantages"], dtype=torch.float32, device=device)


def offpolicy_loss(logits, seq, mask, adv):
    """L = -sum_ij m_ij A_i log pi(y_ij) / sum_ij m_ij  (token mean over the whole group). Log-probs in FP32.
    Returns (loss, per-token logp). An all-empty mask gives a zero loss that still back-propagates."""
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    if not mask.any():
        return logp.sum() * 0.0, logp
    return -(adv[:, None] * logp).masked_select(mask).mean(), logp


def read_cache(cache_dir, need):
    """Groups of one source stream, refused unless the content hash in meta.json matches the file."""
    cache_dir = pathlib.Path(cache_dir); meta = json.load(open(cache_dir / "meta.json"))
    assert sha_file(cache_dir / "groups.jsonl") == meta["content_hash"], f"cache hash mismatch in {cache_dir}"
    groups = [json.loads(l) for l in open(cache_dir / "groups.jsonl")]
    assert len(groups) >= need and [g["step"] for g in groups[:need]] == list(range(1, need + 1))
    return groups[:need], meta


def score(text, gold):
    ok = lambda v: gold is not None and v is not None and math.isclose(v, gold, rel_tol=0.0, abs_tol=1e-6)
    got, strict = extract_final_number(text), extract_strict_hash_number(text)
    return dict(got=got, strict_got=strict, correct=bool(ok(got)), strict_correct=bool(ok(strict)), format_ok=bool("####" in text or "\\boxed" in text),
                # diagnostic only: the text before the model starts another "Question:" block, same rule before and after training
                cutq_correct=bool(ok(extract_final_number(text.split("\nQuestion:")[0]))))


@torch.no_grad()
def greedy_eval(tok, model, eval_rows, eval_ids):
    """The scan's held-out evaluation (greedy, 96 new tokens, batch 8, left padding) keeping EVERY item's text and token ids."""
    was_training = model.training; model.eval()
    ensure_padding_token(tok); side = tok.padding_side; tok.padding_side = "left"
    prompts = [build_prompt(ex, "original") for ex in eval_rows]
    recs = []
    try:
        for i in range(0, len(prompts), EVAL_BATCH):
            enc = tok(prompts[i:i + EVAL_BATCH], padding=True, truncation=True, max_length=1024, return_tensors="pt")
            enc = filter_generate_inputs(model, {k: v.to(next(model.parameters()).device) for k, v in enc.items()})
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for j, text in enumerate(tok.batch_decode(new, skip_special_tokens=True)):
                ids = [t for t in new[j].tolist() if t != tok.pad_token_id]
                gold = extract_final_number(eval_rows[i + j]["orig_answer"])
                recs.append(dict(eval_id=eval_ids[i + j], gold=gold, text=text, output_ids=ids, n_tokens=len(ids), **score(text, gold)))
    finally:
        tok.padding_side = side
        if was_training:
            model.train()
    n = len(recs)
    summ = {k: sum(r[k] for r in recs) / n for k in ("correct", "strict_correct", "format_ok", "cutq_correct")}
    return recs, dict(accuracy=summ["correct"], strict_accuracy=summ["strict_correct"], format_rate=summ["format_ok"],
                      cutq_accuracy=summ["cutq_correct"], mean_tokens=sum(r["n_tokens"] for r in recs) / n, n=n)
