#!/usr/bin/env python3
"""Export the paper's reported terminal scores; no training runtime or network required."""
import argparse
import csv
import io
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "paper_tables.json"
URL = "https://arxiv.org/html/2610.01066v1"
MODEL = "OLMo-2-7B"
FILES = ("checkpoint_runs.csv", "frozen_scores.csv", "input_controls.csv",
         "shared_rollouts.csv", "answer_format.csv")


def score(value):
    try:
        n = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid percentage: {value}") from exc
    if not n.is_finite() or not 0 <= n <= 100:
        raise ValueError(f"percentage outside [0, 100]: {value}")
    return format(n, "f")


def cells(source, table, width, header_rows=2):
    lines = source["tables"][str(table)]["markdown"].splitlines()
    rows = [[x.strip() for x in line.strip().strip("|").split("|")] for line in lines]
    headers = {
        7: ["Checkpoint", "Seed", "Lenient", "Strict", "Format rate"],
        14: ["Checkpoint", "Raw", "GT max", "Random max", "Random mean", "Gain", "Random peak"],
        15: ["ID", "1", "2"] + list("12345678"),
        18: ["Seed", "B", "C", "A", "D", "E"],
        20: ["Seed", "D", "E", "D", "E"],
        23: ["Replicate"] + [rf"{s} → \to {l}" for s, l in
                              [("P1", "P1"), ("P1", "P2"), ("P2", "P1"), ("P2", "P2")]],
    }
    expected = headers.get(table)
    if expected is None:
        expected = (["Checkpoint", "Base"] + list("1234") + ["Max", "n n"] if table in (8, 10, 12)
                    else ["Checkpoint"] + list("12345678") + ["Max"])
    if rows[0 if header_rows == 2 else 2] != expected:
        raise ValueError(f"table {table}: unexpected header or seed order")
    if any(len(row) != width for row in rows[header_rows:]):
        raise ValueError(f"table {table}: unexpected column count")
    return rows[header_rows:]


def checkpoint(label):
    stage, tokens = label.replace(",", "").split()
    if stage not in ("P1", "P2") or not tokens.endswith("B"):
        raise ValueError(f"unknown checkpoint: {label}")
    int(tokens.lstrip("+")[:-1])
    return stage.lower() + "_" + tokens.lstrip("+")


def provenance(table):
    return {"source_table": str(table), "source_url": URL}


def run_id(task, model, ck, reward, seed):
    return f"scan:{task}:{model}:{ck}:{reward}:{seed}"


def build_tables(source=None):
    if source is None:
        source = json.loads(SOURCE.read_text())
    if source["source_url"] != URL or source["arxiv_version"] != "2610.01066v1":
        raise ValueError("expected the pinned paper v1 source")
    runs, frozen, controls, shared, formats = [], [], [], [], []
    for task, gt_table, random_table, count in [("gsm", 8, 9, 37), ("code", 10, 11, 12),
                                               ("lookup", 12, 13, 12)]:
        gt = cells(source, gt_table, 8)
        rnd = cells(source, random_table, 10)
        if len(gt) != count or len(rnd) != count or [x[0] for x in gt] != [x[0] for x in rnd]:
            raise ValueError(f"{task}: checkpoint coverage mismatch")
        for row in gt:
            ck = checkpoint(row[0])
            frozen.append(dict(task=task, model=MODEL, checkpoint=ck, accuracy_pct=score(row[1]),
                               **provenance(gt_table)))
        for reward, table, rows, start, stop in [("ground_truth", gt_table, gt, 2, 6),
                                                ("random", random_table, rnd, 1, 9)]:
            for row in rows:
                values = []
                for ordinal, value in enumerate(row[start:stop], 1):
                    if value in ("—", "–"):
                        continue  # unrun configurations are absent, never zero-filled
                    values.append(Decimal(score(value)))
                    ck, seed = checkpoint(row[0]), 31000 + ordinal
                    runs.append(dict(run_id=run_id(task, MODEL, ck, reward, seed), task=task, model=MODEL,
                                     checkpoint=ck, reward=reward, seed=str(seed), step="500",
                                     accuracy_pct=score(value), **provenance(table)))
                if not values or max(values) != Decimal(score(row[stop])):
                    raise ValueError(f"table {table}: reported maximum does not match seeds")
                if reward == "ground_truth" and len(values) != int(row[-1]):
                    raise ValueError(f"table {table}: reported seed count mismatch")
    extra = cells(source, 14, 7)
    terminals = cells(source, 15, 11, header_rows=3)
    if [x[0] for x in extra] != [x[0] for x in terminals] or [x[0] for x in extra] != list("S" + str(i) for i in range(1, 5)) + list("O" + str(i) for i in range(1, 5)):
        raise ValueError("additional-model coverage mismatch")
    for baseline, row in zip(extra, terminals):
        ck = row[0]
        model = "SmolLM2-1.7B" if ck.startswith("S") else "OLMo-3-7B"
        frozen.append(dict(task="gsm", model=model, checkpoint=ck, accuracy_pct=score(baseline[1]),
                           **provenance(14)))
        for reward, vals, reported_max in [("ground_truth", row[1:3], baseline[2]),
                                           ("random", row[3:11], baseline[3])]:
            if max(Decimal(score(v)) for v in vals) != Decimal(score(reported_max)):
                raise ValueError("additional-model maximum mismatch")
            for i, value in enumerate(vals, 1):
                seed = 31000 + i
                runs.append(dict(run_id=run_id("gsm", model, ck, reward, seed), task="gsm", model=model,
                                 checkpoint=ck, reward=reward, seed=str(seed), step="500",
                                 accuracy_pct=score(value), **provenance(15)))
    for row in cells(source, 18, 6):
        seed = int(row[0])
        for arm, value in zip("BCADE", row[1:]):
            if value in ("—", "–"):
                continue
            reused = run_id("gsm", MODEL, "p2_50B", "random", seed) if arm == "B" and seed <= 31008 else ""
            controls.append(dict(run_id=f"input:{MODEL}:{arm}:{seed}", model=MODEL, arm=arm,
                                 seed=str(seed), step="500", accuracy_pct=score(value),
                                 reuses_run_id=reused, **provenance(18)))
    for row in cells(source, 20, 5, header_rows=3):
        seed = int(row[0])
        for model, arm, value in zip(["Qwen2.5-7B"] * 2 + ["Llama-3.1-8B"] * 2, "DEDE", row[1:]):
            controls.append(dict(run_id=f"input:{model}:{arm}:{seed}", model=model, arm=arm,
                                 seed=str(seed), step="500", accuracy_pct=score(value),
                                 reuses_run_id="", **provenance(20)))
    for row in cells(source, 23, 5):
        replicate = int(row[0])
        for (response_source, learner), value in zip([("P1", "P1"), ("P1", "P2"),
                                                     ("P2", "P1"), ("P2", "P2")], row[1:]):
            shared.append(dict(run_id=f"shared:{response_source}:{learner}:{replicate}",
                               response_source=response_source, learner=learner, replicate=str(replicate),
                               step="500", accuracy_pct=score(value), **provenance(23)))
    for row in cells(source, 7, 5):
        formats.append(dict(checkpoint=checkpoint(row[0]), seed=str(int(row[1])),
                            lenient_accuracy_pct=score(row[2]), strict_accuracy_pct=score(row[3]),
                            format_rate_pct=score(row[4]), **provenance(7)))
    tables = dict(zip(FILES, [runs, frozen, controls, shared, formats]))
    if [len(tables[name]) for name in FILES] != [716, 69, 160, 64, 15]:
        raise ValueError("published result coverage differs from the expected tables")
    ids = [r["run_id"] for rows in [runs, controls, shared] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run identity")
    frozen_ids = [(r["task"], r["model"], r["checkpoint"]) for r in frozen]
    if len(frozen_ids) != len(set(frozen_ids)):
        raise ValueError("duplicate frozen checkpoint")
    by_id = {r["run_id"]: r for r in runs}
    for r in controls:
        if r["reuses_run_id"] and r["accuracy_pct"] != by_id[r["reuses_run_id"]]["accuracy_pct"]:
            raise ValueError("reused run conflicts with checkpoint scan")
    high = {(r["checkpoint"], r["seed"]): r for r in runs
            if r["task"] == "gsm" and r["model"] == MODEL and r["reward"] == "random"
            and Decimal(r["accuracy_pct"]) >= 50}
    if {(r["checkpoint"], r["seed"]) for r in formats} != set(high) or len(high) != len(formats):
        raise ValueError("answer-format diagnostics must cover every high-scoring GSM Random run")
    for r in formats:
        if r["lenient_accuracy_pct"] != high[r["checkpoint"], r["seed"]]["accuracy_pct"]:
            raise ValueError("answer-format score conflicts with terminal score")
    return tables


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def check_csvs(directory=RESULTS):
    tables = build_tables()
    for name, rows in tables.items():
        if (Path(directory) / name).read_bytes() != csv_text(rows).encode():
            raise ValueError(f"{name} differs from the pinned paper tables; regenerate and review the diff")
    return tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--check", action="store_true", help="compare CSV bytes with source tables without writing")
    args = ap.parse_args(argv)
    try:
        if args.check:
            tables = check_csvs(args.out_dir)
        else:
            tables = build_tables()
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for name, rows in tables.items():
                (args.out_dir / name).write_bytes(csv_text(rows).encode())
        for name, rows in tables.items():
            print(f"{name}: {len(rows)} rows")
        print("940 terminal appearances; 8 marked reuses; 932 distinct reported training runs. No training was rerun.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ap.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
