#!/usr/bin/env python3
"""Shared-rollout 2x2: summary tables, the pre-registered paired tests (manifest_main.json -> statistics_plan) and three figures.

    $PROBE_PYTHON summarize_shared_rollouts.py [ROOT]

Unit of analysis = replicate block. Missing terminals are never filled in; paired tests use complete pairs only and say how many.
CPU only. Written and frozen before any formal run finished.
"""
import csv, json, pathlib, random, statistics as st, sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta, binomtest

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PROBE_OUT", "runs") + "/shared_rollout_cross_20260921")
REPS = list(range(51001, 51017)); CELLS = [("S1", "L1"), ("S1", "L2"), ("S2", "L1"), ("S2", "L2")]
NAME = {"S1": "P1-final rollouts", "S2": "P2+5B rollouts", "L1": "learner P1 final", "L2": "learner P2 +5B"}
COL = {"L1": "#1F5FA8", "L2": "#D97B1A"}; INK, MUTED, GRID = "#1a1a19", "#6b6b68", "#e4e3df"   # blue/orange pair, CVD-checked with the dataviz validator


def load():
    runs = {}
    for s, l in CELLS:
        for b in REPS:
            p = ROOT / "runs" / f"{s}_{l}" / f"replicate{b}"
            if (p / "summary.json").exists():
                runs[s, l, b] = json.load(open(p / "summary.json"))
            elif (p / "failure.json").exists():
                runs[s, l, b] = dict(failed=json.load(open(p / "failure.json")))
    return runs


def cp(k, n, a=0.05):
    return (0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k)))


def sign_test(d):
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return dict(positive=pos, negative=neg, zero=len(d) - pos - neg, p=float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else 1.0)


def mcnemar(a, b):
    only_b, only_a = sum(y and not x for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b))
    return dict(only_L2=only_b, only_L1=only_a, both=sum(x and y for x, y in zip(a, b)), p=float(binomtest(only_b, only_b + only_a, 0.5).pvalue) if only_b + only_a else 1.0)


def boot(d, f, n=10000, seed=20260921):
    rng = random.Random(seed); v = sorted(f([rng.choice(d) for _ in d]) for _ in range(n))
    return [v[int(0.025 * n)], v[int(0.975 * n) - 1]]


def main():
    runs = load(); out = ROOT / "summaries"; out.mkdir(exist_ok=True); (ROOT / "figures").mkdir(exist_ok=True)
    good = {k: v for k, v in runs.items() if "failed" not in v}
    rows = [dict(source=s, learner=l, replicate=b, step0=v["step0_accuracy"], **{f"acc_{t}": e["accuracy"] for t, e in sorted(v["evals"].items(), key=lambda kv: int(kv[0]))},
                 terminal=v["terminal_accuracy"], gain=v["gain"], improved=int(v["gain"] > 0.10), at_least_50=int(v["terminal_accuracy"] >= 0.5),
                 collapsed=int(v["terminal_accuracy"] <= 0.05), strict_gain=v["strict_gain"], peak=v["peak_accuracy"], peak_step=v["peak_step"],
                 format_rate_500=v["evals"][max(v["evals"], key=int)]["format_rate"], mean_tokens_500=v["evals"][max(v["evals"], key=int)]["mean_tokens"],
                 cutq_gain=round(v["evals"][max(v["evals"], key=int)]["cutq_accuracy"] - v["evals"]["0"]["cutq_accuracy"], 6) if "0" in v["evals"] else None,
                 attempt=v["attempt"]) for (s, l, b), v in sorted(good.items())]
    for name, data in (("run_summary", rows),):
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r))); w.writeheader(); w.writerows(data)
    cells = []
    for s, l in CELLS:
        r = [x for x in rows if x["source"] == s and x["learner"] == l]; g = [x["gain"] for x in r]; k = sum(x["improved"] for x in r)
        failed = sum(1 for (a, c, b), v in runs.items() if (a, c) == (s, l) and "failed" in v)
        cells.append(dict(source=s, learner=l, planned=16, completed=len(r), failed=failed, missing=16 - len(r) - failed, raw=r[0]["step0"] if r else None,
                          terminals=" ".join(f"{x['terminal']:.3f}" for x in r), gain_mean=round(st.mean(g), 4) if g else None, gain_median=round(st.median(g), 4) if g else None,
                          gain_sd=round(st.pstdev(g), 4) if g else None, gain_max=max(g) if g else None, abs_gain_mean=round(st.mean(abs(x) for x in g), 4) if g else None,
                          improved=k, improved_ci95=[round(x, 3) for x in cp(k, len(r))] if r else None, improved_if_missing_fail=f"{k}/16", improved_if_missing_succeed=f"{k + 16 - len(r)}/16",
                          at_least_50=sum(x["at_least_50"] for x in r), collapsed=sum(x["collapsed"] for x in r), peak_max=max((x["peak"] for x in r), default=None)))
    with open(out / "cell_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)

    gain = {(x["source"], x["learner"], x["replicate"]): x["gain"] for x in rows}; pairs, tests = [], []
    for s in ("S1", "S2"):
        bs = [b for b in REPS if (s, "L1", b) in gain and (s, "L2", b) in gain]
        g1, g2 = [gain[s, "L1", b] for b in bs], [gain[s, "L2", b] for b in bs]
        pairs += [dict(source=s, replicate=b, gain_L1=x, gain_L2=y, d=round(y - x, 6), d_abs=round(abs(y) - abs(x), 6), improved_L1=int(x > 0.10), improved_L2=int(y > 0.10)) for b, x, y in zip(bs, g1, g2)]
        d = [y - x for x, y in zip(g1, g2)]; da = [abs(y) - abs(x) for x, y in zip(g1, g2)]
        tests += [dict(test=f"sign test, gain L2 - L1, {s} row", complete_pairs=len(bs), **sign_test(d), mean_d=st.mean(d) if d else None, median_d=st.median(d) if d else None,
                       mean_d_boot95=boot(d, st.mean) if d else None, median_d_boot95=boot(d, st.median) if d else None),
                  dict(test=f"exact McNemar, improved, {s} row", complete_pairs=len(bs), **mcnemar([x > 0.10 for x in g1], [y > 0.10 for y in g2])),
                  dict(test=f"sign test, |gain| L2 - L1, {s} row (added before the formal runs)", complete_pairs=len(bs), **sign_test(da), mean_d=st.mean(da) if da else None)]
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"]); running = 0.0
    for rank, i in enumerate(order):                                      # Holm step-down over the six primary tests
        running = max(running, min(1.0, (len(tests) - rank) * tests[i]["p"])); tests[i]["p_holm"] = running
    with open(out / "paired_comparisons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]) if pairs else ["source"]); w.writeheader(); w.writerows(pairs)
    expl = []                                                               # exploratory: source effect within a learner, and the interaction
    for l in ("L1", "L2"):
        bs = [b for b in REPS if ("S1", l, b) in gain and ("S2", l, b) in gain]; d = [gain["S2", l, b] - gain["S1", l, b] for b in bs]
        expl.append(dict(contrast=f"gain with P2 rollouts - gain with P1 rollouts, {l}", complete_pairs=len(bs), **sign_test(d), mean_d=st.mean(d) if d else None))
    bs = [b for b in REPS if all((s, l, b) in gain for s, l in CELLS)]; d = [(gain["S2", "L2", b] - gain["S2", "L1", b]) - (gain["S1", "L2", b] - gain["S1", "L1", b]) for b in bs]
    expl.append(dict(contrast="interaction: (L2-L1 | S2) - (L2-L1 | S1)", complete_blocks=len(bs), **sign_test(d), mean_d=st.mean(d) if d else None))
    json.dump(dict(primary_tests_two_sided=tests, exploratory=expl, note="sign tests drop zero differences and test the balance of signs, not a mean; bootstrap resamples whole replicate blocks (10,000, seed 20260921) and is a small-sample interval"),
              open(ROOT / "statistics.json", "w"), indent=1)

    raw = {l: next((x["step0"] for x in rows if x["learner"] == l), None) for l in ("L1", "L2")}
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True, sharey=True)      # 1) every trajectory, rows = source, columns = learner
    for i, s in enumerate(("S1", "S2")):
        for j, l in enumerate(("L1", "L2")):
            a = ax[i][j]
            for x in rows:
                if (x["source"], x["learner"]) == (s, l):
                    ts = sorted(int(k[4:]) for k in x if k.startswith("acc_")); a.plot(ts, [x[f"acc_{t}"] for t in ts], color=COL[l], lw=1.2, alpha=0.55)
            if raw[l] is not None:
                a.axhline(raw[l], color=MUTED, lw=1, ls="--"); a.text(505, raw[l], " raw", fontsize=8, color=MUTED, va="center")
            k = next(c for c in cells if (c["source"], c["learner"]) == (s, l))
            a.set_title(f"{NAME[s]}  →  {NAME[l]}    improved {k['improved']}/{k['completed']}", fontsize=9.5, loc="left", color=INK)
            a.set_ylim(-0.03, 1.03); a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True); a.tick_params(colors=MUTED, length=0)
            for sp in ("top", "right", "left"): a.spines[sp].set_visible(False)
            a.spines["bottom"].set_color(GRID)
    fig.supxlabel("training step (cached rollouts, random reward)", fontsize=9, color=MUTED); fig.supylabel("accuracy, 200 held-out GSM items (greedy)", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "trajectories_2x2.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 4.4), sharey=True)                  # 2) paired gains, one line per replicate
    for a, s in zip(ax, ("S1", "S2")):
        for p in (p for p in pairs if p["source"] == s):
            a.plot([0, 1], [p["gain_L1"], p["gain_L2"]], color=MUTED, lw=1, alpha=0.6, zorder=1)
            a.scatter([0], [p["gain_L1"]], s=36, color=COL["L1"], edgecolor="white", lw=1, zorder=2); a.scatter([1], [p["gain_L2"]], s=36, color=COL["L2"], edgecolor="white", lw=1, zorder=2)
        a.axhline(0, color=INK, lw=1); a.axhline(0.10, color=MUTED, lw=1, ls=":"); a.text(1.32, 0.10, "+10 pts", fontsize=8, color=MUTED, va="center", ha="right")
        a.set_xticks([0, 1]); a.set_xticklabels([NAME["L1"], NAME["L2"]], fontsize=9, color=INK); a.set_xlim(-0.35, 1.35)
        a.set_title(f"same {NAME[s]}, same rewards, same adapter start", fontsize=9.5, loc="left", color=INK)
        a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True); a.tick_params(colors=MUTED, length=0)
        for sp in ("top", "right", "left"): a.spines[sp].set_visible(False)
        a.spines["bottom"].set_color(GRID)
    ax[0].set_ylabel("gain = step-500 accuracy − own step-0 accuracy", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paired_gains.png", dpi=200); plt.close(fig)

    fig, a = plt.subplots(figsize=(7.5, 4.4))                                       # 3) all terminals in the four cells, raw marked separately
    for i, (s, l) in enumerate(CELLS):
        t = [x["terminal"] for x in rows if (x["source"], x["learner"]) == (s, l)]
        a.scatter([i + (k - (len(t) - 1) / 2) * 0.03 for k in range(len(t))], t, s=30, color=COL[l], alpha=0.75, edgecolor="white", lw=0.8, zorder=2)
        if raw[l] is not None:
            a.plot([i - 0.33, i + 0.33], [raw[l]] * 2, color=INK, lw=1.5, zorder=3)
        c = cells[i]; a.text(i, 1.02, f"improved {c['improved']}/{c['completed']}", ha="center", fontsize=9, color=INK)
    a.set_xticks(range(4)); a.set_xticklabels([f"{NAME[s]}\n→ {NAME[l]}" for s, l in CELLS], fontsize=8.5, color=INK); a.set_ylim(-0.03, 1.1)
    a.set_ylabel("step-500 accuracy (black bar = learner's raw score)", fontsize=9, color=MUTED); a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True); a.tick_params(colors=MUTED, length=0)
    for sp in ("top", "right", "left"): a.spines[sp].set_visible(False)
    a.spines["bottom"].set_color(GRID)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "terminals_four_cells.png", dpi=200); plt.close(fig)

    for c in cells:
        print(f"{c['source']}->{c['learner']}: {c['completed']}/16 done, raw {c['raw']}, gain mean {c['gain_mean']} median {c['gain_median']} sd {c['gain_sd']} |gain| mean {c['abs_gain_mean']}, improved {c['improved']}, collapsed {c['collapsed']}, >=50% {c['at_least_50']}, peak max {c['peak_max']}")
    for t in tests:
        print(f"  {t['test']}: n={t['complete_pairs']} " + " ".join(f"{k}={t[k]}" for k in ("positive", "negative", "zero", "only_L2", "only_L1") if k in t) + f" p={t['p']:.4g} holm={t['p_holm']:.4g}")


if __name__ == "__main__":
    main()
