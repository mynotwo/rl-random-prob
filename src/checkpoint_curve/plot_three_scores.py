#!/usr/bin/env python3
"""Checkpoint-curve figure: 37 checkpoints equally spaced, P1 | P2.
Blue = frozen greedy, green = max over GT seeds of the step-500 score, orange = same for Random
(--no_random leaves it out while those runs are still going). Faint dots are the individual seeds;
a hollow max marker means fewer than the target number of seeds is present in this directory.

    python plot_three_scores.py RUN_DIR --out three_scores.png [--no_random] [--points_json out.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPO / "configs" / "olmo2_checkpoints.json"
BLUE, GREEN, ORANGE = "#1F5FA8", "#127A45", "#B07306"


def collect(run_dir, layout="gsm", task=None, manifest=MANIFEST):
    """gsm layout: <run_dir>/<tag>/rlvr_*.jsonl + frozen.json (the 37-point checkpoint manifest).
    supp layout: <run_dir>/<task>/rlvr_*..._<tag>_lr..jsonl plus the identical-protocol runs listed
    under manifest.json["reused"]; checkpoints and seed targets come from that manifest."""
    if layout == "gsm":
        m = json.load(open(manifest)); cks = [(c["tag"], c["stage"], c["token_label_B"]) for c in m["checkpoints"]]
        files = glob.glob(f"{run_dir}/*/rlvr_*.jsonl"); n_gt, n_rand = len(m["target_gt_seeds"]), len(m["target_random_seeds"])
    else:
        m = json.load(open(f"{run_dir}/manifest.json")); cks = [(c["tag"], c["stage"], c["tokens_b"]) for c in m["checkpoints"]]
        files = glob.glob(f"{run_dir}/{task}/rlvr_*.jsonl") + [r["out"] for r in m["reused"] if r["key"]["task"] == task]
        n_gt, n_rand = (2 if task == "twohop" else len(m["gt_seeds"])), len(m["random_seeds"])
    term = {"ground_truth": collections.defaultdict(dict), "random": collections.defaultdict(dict)}
    before = {}
    for f in files:
        recs = [json.loads(l) for l in open(f) if l.strip()]
        after = [r for r in recs if r["phase"] == "after"]
        if not after:
            continue
        tag = os.path.basename(os.path.dirname(f)) if layout == "gsm" else re.search(r"_(p[12]_\d+B)_lr", f).group(1)
        meta = recs[0]
        term[meta["reward_mode"]][tag][meta["seed"]] = after[0]["accuracy"]
        before[tag] = next(r for r in recs if r["phase"] == "before")["accuracy"]
    pts = []
    for tag, stage, tokens in cks:
        fz = f"{run_dir}/{tag}/frozen.json"
        frozen = json.load(open(fz))["summary"]["greedy_lenient_accuracy"] if os.path.exists(fz) else before.get(tag)
        pts.append(dict(tag=tag, stage=stage, tokens=tokens, frozen=frozen,
                        gt=term["ground_truth"].get(tag, {}), random=term["random"].get(tag, {})))
    return pts, n_gt, n_rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_random", action="store_true")
    ap.add_argument("--points_json")
    ap.add_argument("--layout", choices=["gsm", "supp"], default="gsm")
    ap.add_argument("--task", default=None, help="supp layout: code or twohop")
    ap.add_argument("--chance", type=float, default=0.0, help="draw a chance line (twohop 0.25)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    pts, n_gt, n_rand = collect(args.run_dir, args.layout, args.task)
    if args.points_json:
        json.dump(pts, open(args.points_json, "w"), indent=1)
    plot_points(pts, n_gt, n_rand, args)


def plot_points(pts, n_gt, n_rand, args):
    """Draw collected points; also used by the CSV-only published-results entry point."""
    x = list(range(len(pts)))
    n_p1 = sum(p["stage"] == 1 for p in pts)
    fig, ax = plt.subplots(figsize=(14 if len(pts) > 20 else 9, 4.6))
    ax.axvspan(n_p1 - 0.5, len(pts) - 0.5, color="#000", alpha=0.05, lw=0)
    ax.axvline(n_p1 - 0.5, color="#333", lw=1)
    series = [("gt", GREEN, "s", f"GT max-of-{n_gt} (step 500)", n_gt)]
    if not args.no_random:
        series.append(("random", ORANGE, "^", f"Random max-of-{n_rand} (step 500)", n_rand))
    for key, color, mk, label, target in series:
        for i, p in enumerate(pts):
            ax.scatter([i] * len(p[key]), list(p[key].values()), s=10, color=color, alpha=0.35, lw=0, zorder=2)
        xs = [i for i, p in enumerate(pts) if p[key]]
        ys = [max(pts[i][key].values()) for i in xs]
        ax.plot(xs, ys, "-", color=color, lw=2, zorder=3, label=getattr(args, f"{key}_label", label))
        for i, y in zip(xs, ys):
            full = len(pts[i][key]) >= pts[i].get(f"{key}_target", target)
            ax.scatter([i], [y], s=42, marker=mk, zorder=4, facecolor=color if full else "white", edgecolor=color, lw=1.6)
    ax.plot(x, [p["frozen"] for p in pts], "-o", color=BLUE, lw=2, ms=6, zorder=3, label="frozen greedy")
    ax.set_xticks(x)
    ax.set_xticklabels([("+" if p["stage"] == 2 else "") + f"{p['tokens']}B" for p in pts], rotation=60, ha="right", fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.6, len(pts) - 0.4)
    ax.set_ylabel(getattr(args, "ylabel", "lenient greedy accuracy (200 items)"))
    ax.text((n_p1 - 1) / 2, 1.04, "P1 · stage 1 (tokens seen)", ha="center", fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(n_p1 + (len(pts) - n_p1 - 1) / 2, 1.04, "P2 · stage 2 (+tokens)", ha="center", fontsize=9, transform=ax.get_xaxis_transform())
    if args.chance > 0:
        ax.axhline(args.chance, color="#888", lw=0.9, ls="--", label=f"chance {args.chance}")
    if args.title:
        ax.set_title(args.title, loc="left", fontsize=11, fontweight="bold", pad=22)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left" if args.layout == "gsm" else "lower right", fontsize=8, frameon=False, title="faint dots = seeds · hollow = fewer than target seeds here", title_fontsize=7)
    caption = getattr(args, "caption", None)
    if caption:
        fig.text(0.01, 0.005, caption, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.06 if caption else 0, 1, 1))
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"{len(pts)} checkpoints -> {args.out}")


if __name__ == "__main__":
    main()
