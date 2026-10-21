#!/usr/bin/env python3
"""Offline audit of the shared-rollout experiment: integrity of what the learners consumed, and the correctness sidecar.

    $PROBE_PYTHON audit_shared_rollouts.py [ROOT]          # CPU only; writes ROOT/audits/

The sidecar (correctness_audit.jsonl) is the ONLY place gold answers meet the cached rollouts. It is written after the fact
and nothing in the training path reads it. Reward-misalignment columns are plain counts from rewards x correctness
(rewarded-but-wrong, unrewarded-but-correct, advantage mass on correct answers); they are the ingredients of Chen et al.
section 5, not a claim to reproduce their exact definition.
"""
import json, pathlib, sys
import os

import sr_common as C

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PROBE_OUT", "runs") + "/shared_rollout_cross_20260921")


def main():
    out = ROOT / "audits"; out.mkdir(exist_ok=True)
    train_rows, _, train_ids, _ = C.load_split(); gold = {i: C.extract_final_number(r["orig_answer"]) for i, r in zip(train_ids, train_rows)}
    problems, streams, side = [], {}, open(out / "correctness_audit.jsonl", "w")
    for d in sorted((ROOT / "rollouts").glob("S*/replicate*")):
        if not (d / "meta.json").exists():
            continue
        s, b = d.parent.name, int(d.name.replace("replicate", "")); meta = json.load(open(d / "meta.json"))
        if C.sha_file(d / "groups.jsonl") != meta["content_hash"]:
            problems.append(f"{s} {b}: cache content hash mismatch")
        groups = [json.loads(l) for l in open(d / "groups.jsonl")]
        if any(C.group_hash(g) != g["group_hash"] for g in groups):
            problems.append(f"{s} {b}: a group does not match its hash")
        if [g["rewards"] for g in groups] != C.reward_table(b).int().tolist()[:len(groups)]:
            problems.append(f"{s} {b}: rewards differ from the replicate's table")
        streams[s, b] = [g["group_hash"] for g in groups]
        nc = ng = fr = mr = 0; mass = 0.0
        for g in groups:
            ok = [C.score(t, gold[g["train_item_id"]])["correct"] for t in g["response_text"]]
            rec = dict(source=s, replicate=b, step=g["step"], cache_key=g["group_hash"], correct=ok, n_correct=sum(ok), rewards=g["rewards"],
                       rewarded_but_wrong=sum(r and not c for r, c in zip(g["rewards"], ok)), unrewarded_but_correct=sum(c and not r for r, c in zip(g["rewards"], ok)),
                       advantage_on_correct=sum(a for a, c in zip(g["advantages"], ok) if c))
            side.write(json.dumps(rec) + "\n"); nc += sum(ok); ng += len(ok); fr += rec["rewarded_but_wrong"]; mr += rec["unrewarded_but_correct"]; mass += rec["advantage_on_correct"]
        meta.update(rollout_accuracy=nc / ng, rewarded_but_wrong=fr, unrewarded_but_correct=mr, advantage_on_correct_sum=mass,
                    mean_tokens=sum(sum(g["generated_lengths"]) for g in groups) / ng, eos_rate=sum(r[-1] == meta["eos_token_id"] for g in groups for r in g["response_input_ids"]) / ng)
        streams[s, b, "meta"] = meta
    side.close()
    consumed, inits, step0 = {}, {}, {}
    for d in sorted((ROOT / "runs").glob("S*_L*/replicate*")):
        if not (d / "summary.json").exists():
            continue
        (s, l), b = d.parent.name.split("_"), int(d.name.replace("replicate", "")); sm = json.load(open(d / "summary.json"))
        keys = [json.loads(x)["cache_key"] for x in open(d / "train_metrics.jsonl") if '"phase": "train"' in x]
        if keys != streams.get((s, b), [])[:len(keys)]:
            problems.append(f"{s}->{l} {b}: consumed cache keys differ from the source stream")
        consumed[s, l, b] = keys; inits.setdefault(b, set()).add(sm["adapter_init_hash"]); step0.setdefault(l, set()).add(sm["step0_accuracy"])
        if not sm["base_frozen"]:
            problems.append(f"{s}->{l} {b}: base tensors changed")
    for (s, l, b), keys in consumed.items():
        other = consumed.get((s, "L2" if l == "L1" else "L1", b))
        if other is not None and other != keys:
            problems.append(f"{s} {b}: the two learners did not consume the same groups")
    problems += [f"replicate {b}: {len(h)} different adapter starts" for b, h in inits.items() if len(h) != 1]
    problems += [f"learner {l}: step-0 accuracy is not unique across runs: {sorted(v)}" for l, v in step0.items() if len(v) != 1]
    rep = dict(problems=problems, runs_checked=len(consumed), streams_checked=sum(1 for k in streams if len(k) == 2), step0_accuracy={l: sorted(v) for l, v in step0.items()},
               streams=[dict(source=k[0], replicate=k[1], **{f: v[f] for f in ("rollout_accuracy", "mean_tokens", "eos_rate", "rewarded_but_wrong", "unrewarded_but_correct", "advantage_on_correct_sum", "seconds_generate")})
                        for k, v in streams.items() if len(k) == 3])
    json.dump(rep, open(out / "audit.json", "w"), indent=1)
    print(f"{rep['streams_checked']} streams, {rep['runs_checked']} runs checked; problems: {problems or 'none'}")
    for s in ("S1", "S2"):
        a = [x for x in rep["streams"] if x["source"] == s]
        if a:
            print(f"  {s}: rollout accuracy {sum(x['rollout_accuracy'] for x in a) / len(a):.3f}, mean tokens {sum(x['mean_tokens'] for x in a) / len(a):.1f}, EOS rate {sum(x['eos_rate'] for x in a) / len(a):.2f}")


if __name__ == "__main__":
    main()
