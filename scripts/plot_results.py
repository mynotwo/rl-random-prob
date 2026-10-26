#!/usr/bin/env python3
"""Redraw published terminal results from checked CSVs, without torch, models or a GPU."""
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from export_paper_results import MODEL, RESULTS, ROOT, SOURCE, URL, check_csvs, csv_text

COLORS = ["#1F5FA8", "#D97B1A", "#0E6E3E", "#9265A8", "#6B6B68"]
NOTE = "Paper v1 terminal scores at step 500. Maxima are best-of-N, not means or a capability ceiling.\nCheckpoints are equally spaced in order, not on a token-scale axis; lines guide the eye."


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def values(rows):
    return [float(r["accuracy_pct"]) for r in rows]


def save(fig, path):
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + extension), dpi=180)
    plt.close(fig)


def checkpoint_plots(runs, frozen, out):
    renderer = module("checkpoint_plot", ROOT / "src/checkpoint_curve/plot_three_scores.py")
    for task in ("gsm", "code", "lookup"):
        pts = []
        for r in frozen:
            if r["model"] != MODEL or r["task"] != task:
                continue
            ck = r["checkpoint"]
            stage = int(ck[1])
            p = dict(tag=ck, stage=stage, tokens=int(ck.split("_")[1][:-1]),
                     frozen=float(r["accuracy_pct"]) / 100)
            for name, reward in [("gt", "ground_truth"), ("random", "random")]:
                selected = [x for x in runs if x["model"] == MODEL and x["task"] == task
                            and x["checkpoint"] == ck and x["reward"] == reward]
                p[name] = {int(x["seed"]): float(x["accuracy_pct"]) / 100 for x in selected}
                p[name + "_target"] = len(selected)
            pts.append(p)
        gt_n, rand_n = (4, 8) if task != "lookup" else (2, 8)
        for ext in ("png", "pdf"):
            args = SimpleNamespace(out=out / f"olmo2_{task}.{ext}", no_random=False, layout="gsm",
                                   chance=0.25 if task == "lookup" else 0,
                                   title=f"OLMo-2 · {task} · reported outcomes",
                                   ylabel=f"Held-out {task} accuracy (200 items)", caption=NOTE)
            if task == "code":
                args.gt_label = "GT terminal max (P1 n=1, P2 n=4)"
                args.random_label = "Random terminal max (P1 n=2, P2 n=8)"
            renderer.plot_points(pts, gt_n, rand_n, args)


def extra_model_plot(runs, frozen, out):
    renderer = module("cross_model_plot", ROOT / "src/cross_model/summarize.py")
    rows = []
    for r in frozen:
        if r["model"] == MODEL:
            continue
        p = dict(checkpoint=r["checkpoint"], model=("OLMo 3-7B" if r["model"] == "OLMo-3-7B" else r["model"]), raw_greedy=float(r["accuracy_pct"]) / 100)
        for key, reward in [("gt", "ground_truth"), ("random", "random")]:
            terminal = [float(x["accuracy_pct"]) / 100 for x in runs
                        if x["model"] == r["model"] and x["checkpoint"] == r["checkpoint"]
                        and x["reward"] == reward]
            p[key + "_max"] = max(terminal)
            p[key + "_terminals"] = " ".join(str(x) for x in terminal)
        rows.append(p)
    for ext in ("png", "pdf"):
        renderer.plot_checkpoints(rows, out / f"additional_models.{ext}")


def control_plot(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, model in zip(axes, [MODEL, "Qwen2.5-7B", "Llama-3.1-8B"]):
        arms = list("BCADE") if model == MODEL else list("DE")
        for i, arm in enumerate(arms):
            v = values([r for r in rows if r["model"] == model and r["arm"] == arm])
            xs = [i + (j - (len(v) - 1) / 2) * 0.018 for j in range(len(v))]
            ax.scatter(xs, v, s=17, color=COLORS["BCADE".index(arm)], alpha=0.65)
            ax.scatter([i], [max(v)], s=55, marker="D", color=COLORS["BCADE".index(arm)], edgecolors="black", linewidths=0.6)
            ax.text(i, 103, f"n={len(v)}", ha="center", fontsize=8)
        ax.set_xticks(range(len(arms)), arms)
        ax.set_ylim(-2, 111)
        ax.set_title(model)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Terminal GSM accuracy (%) · 200 items")
    fig.text(0.015, 0.01, "Dots = all reported seeds; diamonds = terminal maxima. B: GSM; C: shuffled GSM; A: MMLU; "
             "D: tokens + template; E: raw tokens.\nDifferent seed counts; points are terminal outcomes, not training trajectories.", fontsize=7)
    save(fig, out / "input_controls")


def shared_plot(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, response_source in zip(axes, ("P1", "P2")):
        for replicate in sorted({r["replicate"] for r in rows}):
            pair = {r["learner"]: float(r["accuracy_pct"]) for r in rows
                    if r["response_source"] == response_source and r["replicate"] == replicate}
            ax.plot([0, 1], [pair["P1"], pair["P2"]], color="#888", alpha=0.45, lw=0.8)
            ax.scatter([0, 1], [pair["P1"], pair["P2"]], color=COLORS[:2], s=22, zorder=3)
        ax.set_xticks([0, 1], ["P1 learner", "P2 learner"])
        ax.set_title(f"{response_source} response source · 16 paired blocks")
        ax.set_ylim(-2, 102)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Terminal GSM accuracy (%) · 200 items")
    fig.text(0.015, 0.01, "Each line joins the same replicate's terminal outcomes. Source → learner is preserved. "
             "Reported initial scores differ: P1 2.5%, P2 13.5%.", fontsize=7)
    save(fig, out / "shared_rollouts")


def format_plot(rows, out):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for offset, key, label, color in [(-0.2, "lenient_accuracy_pct", "Lenient", COLORS[0]),
                                     (0.2, "strict_accuracy_pct", "Strict", COLORS[1])]:
        ax.bar([i + offset for i in range(len(rows))], [float(r[key]) for r in rows],
               width=0.38, label=label, color=color)
    ax.set_xticks(range(len(rows)), [f"{r['checkpoint'].replace('p2_', '+')}\n{r['seed']}" for r in rows],
                  rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Terminal accuracy (%) · 200 items")
    ax.set_title("All 15 GSM Random runs with lenient terminal accuracy ≥ 50%")
    ax.legend(frameon=False)
    fig.text(0.015, 0.01, "Selected diagnostic subset from Table 7, not all Random runs. "
             "Strict extraction also requires the requested answer-line structure.", fontsize=7)
    save(fig, out / "answer_format")


def summary_rows(runs):
    rows = []
    keys = list(dict.fromkeys((r["task"], r["model"], r["checkpoint"], r["reward"]) for r in runs))
    for task, model, ck, reward in keys:
        v = values([r for r in runs if (r["task"], r["model"], r["checkpoint"], r["reward"])
                    == (task, model, ck, reward)])
        rows.append(dict(task=task, model=model, checkpoint=ck, reward=reward, n=len(v),
                         terminal_max_pct=max(v), terminal_mean_pct=statistics.mean(v),
                         terminal_median_pct=statistics.median(v)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/paper")
    args = ap.parse_args(argv)
    try:
        # Check the actual input CSVs before drawing; never substitute missing scores or redownload data.
        check_csvs(args.results_dir)
        data = {name: read_csv(args.results_dir / name) for name in
                ("checkpoint_runs.csv", "frozen_scores.csv", "input_controls.csv", "shared_rollouts.csv", "answer_format.csv")}
        args.out_dir.mkdir(parents=True, exist_ok=True)
        runs, frozen = data["checkpoint_runs.csv"], data["frozen_scores.csv"]
        checkpoint_plots(runs, frozen, args.out_dir)
        extra_model_plot(runs, frozen, args.out_dir)
        control_plot(data["input_controls.csv"], args.out_dir)
        shared_plot(data["shared_rollouts.csv"], args.out_dir)
        format_plot(data["answer_format.csv"], args.out_dir)
        (args.out_dir / "checkpoint_summary.csv").write_bytes(csv_text(summary_rows(runs)).encode())
        manifest = {"scope": "Redraw of reported paper v1 results; no model training or evaluation rerun",
                    "paper_source": URL, "source_tables_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    "python": platform.python_version(), "matplotlib": matplotlib.__version__,
                    "torch_imported": "torch" in sys.modules,
                    "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(args.results_dir.glob("*.csv"))},
                    "outputs": [f"{name}.{ext}" for name in ("olmo2_gsm", "olmo2_code", "olmo2_lookup",
                                 "additional_models", "input_controls", "shared_rollouts", "answer_format")
                                for ext in ("png", "pdf")]}
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"7 figures (PNG + PDF) and checkpoint_summary.csv -> {args.out_dir}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        ap.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
