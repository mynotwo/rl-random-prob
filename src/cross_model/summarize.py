#!/usr/bin/env python3
"""SmolLM2 / OLMo 3 supplement: run_summary.csv, checkpoint_summary.csv and the two-panel
three-line figure (Raw / GT max / Random max, every seed's step-500 score as a faint dot).

    $PROBE_PYTHON cross_model/summarize.py [RUN_DIR]        # writes RUN_DIR/summaries/ and RUN_DIR/figures/
"""
import csv, glob, json, pathlib, re, statistics as st, sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

RUN = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PROBE_OUT", "runs") + "/smollm2_olmo3_supplement_20260920")
# id, x tick label, model
CKPTS = [("S1", "S1\n0.26T", "SmolLM2-1.7B"), ("S2", "S2\n5.77T", "SmolLM2-1.7B"), ("S3", "S3\n9.96T", "SmolLM2-1.7B"), ("S4", "S4\n10.75T", "SmolLM2-1.7B"),
         ("O1", "O1\nP1 34B", "OLMo 3-7B"), ("O2", "O2\nP1 end\n5.93T", "OLMo 3-7B"), ("O3", "O3\nP2 +4.2B", "OLMo 3-7B"), ("O4", "O4\nP2 +100B", "OLMo 3-7B")]
# Raw / GT / Random. Blue as in the OLMo 2 figure; green and orange re-stepped so the pair passes the CVD check
# (the OLMo 2 figure's #127A45 / #B07306 fail it, protan dE 5.9). Marker shape + direct labels are the secondary encoding.
SERIES = [("raw", "Raw (frozen, greedy)", "#1F5FA8", "o"), ("gt", "GT RL, max of 2 seeds", "#0E6E3E", "s"), ("random", "Random RL, max of 8 seeds", "#D97B1A", "^")]
INK, MUTED, GRID = "#1a1a19", "#6b6b68", "#e4e3df"


def first_answer_acc(records):
    """Raw accuracy on the text before the model restarts a new block (it answers, then rambles on to 96 tokens)."""
    from probe_checkpoints import extract_final_number

    ok = lambda t, g: (lambda v: v is not None and g is not None and abs(v - g) < 1e-6)(extract_final_number(re.split(r"\n\s*\n|\nQuestion:|\nSolve the", t, maxsplit=1)[0]))
    return sum(ok(r["greedy"], r["gold"]) for r in records) / len(records)


def runs(cid, mode):
    out = []
    for p in sorted(glob.glob(f"{RUN}/runs/{cid}/{mode}/seed*/*.jsonl")):
        recs = [json.loads(l) for l in open(p)]
        if not any(r.get("phase") == "after" for r in recs):
            continue                                    # no terminal eval -> missing, never filled with zero
        ev = {r["step"]: r for r in recs if r.get("phase") == "eval"}
        acc = {s: e["accuracy"] for s, e in ev.items()}
        b = next(r for r in recs if r.get("phase") == "before")
        pk = max(acc, key=acc.get)
        out.append(dict(checkpoint=cid, reward=mode, seed=int(re.search(r"seed(\d+)", p).group(1)), step0=b["accuracy"],
                        **{f"acc_{s}": acc[s] for s in sorted(acc)}, terminal=acc[500], delta=round(acc[500] - b["accuracy"], 4),
                        strict_terminal=ev[500].get("strict_accuracy"), format_rate_500=ev[500]["format_rate"], peak=acc[pk], peak_step=pk,
                        file=str(pathlib.Path(p).relative_to(RUN))))
    return out


def write_csv(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def main():
    (RUN / "summaries").mkdir(exist_ok=True); (RUN / "figures").mkdir(exist_ok=True)
    allruns, ck = [], []
    for cid, _, model in CKPTS:
        fz = json.load(open(RUN / "frozen" / cid / "frozen.json"))
        g, r = runs(cid, "gt"), runs(cid, "random")
        allruns += g + r
        gt, rt = [x["terminal"] for x in g], [x["terminal"] for x in r]
        ck.append(dict(checkpoint=cid, model=model, raw_greedy=fz["summary"]["greedy_lenient_accuracy"], raw_greedy_first_answer=round(first_answer_acc(fz["records"]), 4),
                       raw_sample_acc=fz["summary"]["sample_lenient_accuracy"], raw_pass_at_8=fz["summary"]["pass_at_8_lenient"],
                       gt_n=len(gt), gt_terminals=" ".join(f"{x:.3f}" for x in gt), gt_max=max(gt), gt_mean=round(st.mean(gt), 4),
                       random_n=len(rt), random_terminals=" ".join(f"{x:.3f}" for x in rt), random_max=max(rt), random_mean=round(st.mean(rt), 4),
                       random_median=round(st.median(rt), 4), random_sd=round(st.pstdev(rt), 4),
                       random_up_gt_10pp=sum(x["delta"] > 0.10 for x in r), random_down_gt_10pp=sum(x["delta"] < -0.10 for x in r),
                       random_peak_max=max(x["peak"] for x in r), random_peaked_then_fell=sum(x["peak"] - x["terminal"] > 0.10 for x in r),
                       gt_peak_max=max(x["peak"] for x in g)))
    write_csv(RUN / "summaries" / "run_summary.csv", allruns)
    write_csv(RUN / "summaries" / "checkpoint_summary.csv", ck)

    plot_checkpoints(ck, RUN / "figures" / "three_lines_smollm2_olmo3.png")
    for c in ck:
        print(f"{c['checkpoint']}: raw {c['raw_greedy']:.3f} (first-answer {c['raw_greedy_first_answer']:.3f}) | GT max {c['gt_max']:.3f} | Random max {c['random_max']:.3f} mean {c['random_mean']:.3f} SD {c['random_sd']:.3f} up {c['random_up_gt_10pp']}/8 down {c['random_down_gt_10pp']}/8")


def plot_checkpoints(ck, out):
    """Draw summary rows without loading model runtimes or frozen response records."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, model in zip(axes, ("SmolLM2-1.7B", "OLMo 3-7B")):
        rows = [c for c in ck if c["model"] == model]; xs = range(len(rows))
        for key, label, color, marker in SERIES:
            ys = [c["raw_greedy"] if key == "raw" else c[f"{key}_max"] for c in rows]
            if key != "raw":                            # every seed's step-500 score, jittered slightly so equal values stay visible
                for x, c in zip(xs, rows):
                    t = [float(v) for v in c[f"{key}_terminals"].split()]
                    ax.scatter([x + (i - (len(t) - 1) / 2) * 0.035 for i in range(len(t))], t, s=14, color=color, alpha=0.35, linewidths=0, zorder=2)
            ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, mec="white", mew=1.5, label=label, zorder=3)
        ends = sorted(((c_, (rows[-1]["raw_greedy"] if k == "raw" else rows[-1][f"{k}_max"]), l.split(",")[0].split(" (")[0]) for k, l, c_, _ in SERIES), key=lambda e: e[1])
        ypos = []
        for color, y, text in ends:                     # direct labels at the last point, pushed apart when they would collide
            y = max(y, ypos[-1] + 0.055) if ypos else y; ypos.append(y)
            ax.annotate(text, (len(rows) - 1, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_xticks(list(xs)); ax.set_xticklabels([next(t for i, t, _ in CKPTS if i == c["checkpoint"]) for c in rows], fontsize=9, color=INK)
        ax.set_xlim(-0.4, len(rows) - 1 + 0.95); ax.set_ylim(-0.03, 1.05)
        ax.set_title(model, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0)
        if model.startswith("OLMo"):
            ax.axvline(1.5, color=MUTED, lw=1, ls=":"); ax.text(1.53, 1.03, "mid-training starts", fontsize=8, color=MUTED, va="top")
    axes[0].set_ylabel("accuracy on the 200 held-out synthetic-GSM items", fontsize=9, color=MUTED)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK)
    fig.text(0.01, 0.005, "Lines: Raw = frozen greedy accuracy; GT / Random = highest step-500 score over seeds (GT n=2, Random n=8, different n: not a comparison of means). "
             "Dots: every seed's step-500 score.\nCheckpoints are equally spaced, not to scale; lines between them only guide the eye.", fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200); plt.close(fig); print(out)


if __name__ == "__main__":
    main()
