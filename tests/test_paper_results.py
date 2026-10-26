"""Paper values and failure paths, independent of model/training dependencies."""
import copy
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_paper_results import FILES, MODEL, SOURCE, build_tables, check_csvs, csv_text, score


class PaperResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = build_tables()

    def test_all_published_rows_and_reuse_are_accounted_for(self):
        self.assertEqual([len(self.tables[k]) for k in FILES], [716, 69, 160, 64, 15])
        appearances = self.tables[FILES[0]] + self.tables[FILES[2]] + self.tables[FILES[3]]
        identities = {r.get("reuses_run_id") or r["run_id"] for r in appearances}
        self.assertEqual(len(appearances), 940)
        self.assertEqual(len(identities), 932)
        self.assertEqual(sum(bool(r["reuses_run_id"]) for r in self.tables[FILES[2]]), 8)
        self.assertTrue(all(r["step"] == "500" for r in appearances))

    def test_gsm_checkpoints_match_the_existing_protocol_manifest(self):
        expected = json.loads((ROOT / "configs/olmo2_checkpoints.json").read_text())["checkpoints"]
        actual = [r["checkpoint"] for r in self.tables[FILES[1]] if r["model"] == MODEL and r["task"] == "gsm"]
        self.assertEqual(actual, [c["tag"] for c in expected])

    def select(self, **fields):
        return [r for r in self.tables[FILES[0]] if all(r[k] == v for k, v in fields.items())]

    def test_paper_matched_baseline_example(self):
        frozen = {(r["task"], r["checkpoint"]): r["accuracy_pct"]
                  for r in self.tables[FILES[1]] if r["model"] == MODEL}
        for ck, maximum in [("p1_462B", 8.5), ("p1_839B", 55.0)]:
            self.assertEqual(frozen["gsm", ck], "3.5")
            rows = self.select(task="gsm", model=MODEL, checkpoint=ck, reward="ground_truth")
            self.assertEqual(max(float(r["accuracy_pct"]) for r in rows), maximum)

    def test_terminal_successes_are_not_scheduled_peaks(self):
        rows = self.select(task="gsm", model=MODEL, reward="random")
        for prefix, expected_n, expected_high in [("p1_", 200, 0), ("p2_", 96, 15)]:
            selected = [r for r in rows if r["checkpoint"].startswith(prefix)]
            self.assertEqual(len(selected), expected_n)
            self.assertEqual(sum(Decimal(r["accuracy_pct"]) >= 50 for r in selected), expected_high)
        self.assertTrue(all(Decimal(r["strict_accuracy_pct"]) < 50 for r in self.tables[FILES[4]]))

    def test_unrun_code_seeds_are_absent_and_real_zeros_remain(self):
        gt = self.select(task="code", model=MODEL, checkpoint="p1_5B", reward="ground_truth")
        random = self.select(task="code", model=MODEL, checkpoint="p1_5B", reward="random")
        self.assertEqual([(r["seed"], r["accuracy_pct"]) for r in gt], [("31001", "0.0")])
        self.assertEqual(len(random), 2)
        self.assertEqual(Counter(r["reward"] for r in self.select(task="code", model=MODEL)),
                         {"ground_truth": 24, "random": 48})

    def test_additional_models_and_control_means_use_all_seeds(self):
        o4 = self.select(model="OLMo-3-7B", checkpoint="O4", reward="random")
        self.assertEqual(sum(Decimal(r["accuracy_pct"]) for r in o4) / 8, Decimal("47.6875"))
        controls = self.tables[FILES[2]]
        for arm, n, expected in [("B", 32, "13.984375"), ("D", 32, "17.5"), ("E", 32, "20.359375")]:
            rows = [r for r in controls if r["model"] == MODEL and r["arm"] == arm]
            self.assertEqual(len(rows), n)
            self.assertEqual(sum(Decimal(r["accuracy_pct"]) for r in rows) / n, Decimal(expected))

    def test_shared_rollout_pairing_and_reported_endpoints(self):
        rows = self.tables[FILES[3]]
        for replicate in map(str, range(51001, 51017)):
            self.assertEqual({(r["response_source"], r["learner"]) for r in rows if r["replicate"] == replicate},
                             {("P1", "P1"), ("P1", "P2"), ("P2", "P1"), ("P2", "P2")})
        p1p2 = [Decimal(r["accuracy_pct"]) for r in rows
                if (r["response_source"], r["learner"]) == ("P1", "P2")]
        self.assertEqual(sum(p1p2) / 16, 18)
        self.assertEqual(max(p1p2), 80)

    def test_nonfinite_out_of_range_and_missing_scores_fail(self):
        for value in ("NaN", "Infinity", "-1", "100.1", "", "—"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                score(value)

    def test_source_column_order_maxima_and_reuse_conflicts_fail(self):
        source = json.loads(SOURCE.read_text())
        mutations = [
            ("8", "| Base | 1 | 2 |", "| Base | 2 | 1 |"),
            ("8", "| P1 5B | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 |", "| P1 5B | 0.0 | 100.0 | 0.0 | 0.0 | 0.5 | 0.5 |"),
            ("18", "| 31001 | 23.5 |", "| 31001 | 23.0 |"),
            ("23", "| 51002 |", "| 51001 |"),
            ("9", "| P1 34B |", "| P1 5B |"),
        ]
        for table, old, new in mutations:
            with self.subTest(table=table):
                changed = copy.deepcopy(source)
                self.assertIn(old, changed["tables"][table]["markdown"])
                changed["tables"][table]["markdown"] = changed["tables"][table]["markdown"].replace(old, new, 1)
                with self.assertRaises(ValueError):
                    build_tables(changed)

    def test_csv_roundtrip_and_missing_duplicate_conflicting_rows_fail(self):
        self.assertEqual(check_csvs(), self.tables)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, rows in self.tables.items():
                (path / name).write_bytes(csv_text(rows).encode())
            self.assertEqual(check_csvs(path), self.tables)
            target = path / FILES[0]
            original = target.read_text()
            lines = original.splitlines(keepends=True)
            for bad in ("".join(lines[:-1]), original + lines[-1], original.replace("0.5", "NaN", 1)):
                target.write_text(bad)
                with self.assertRaises(ValueError):
                    check_csvs(path)
            parsed = list(csv.DictReader(io.StringIO(original)))
            self.assertEqual(parsed, self.tables[FILES[0]])

    def test_public_cli_generates_all_figures_without_torch(self):
        self.assertIsNone(importlib.util.find_spec("torch"), "run plot tests with requirements-plot.txt only")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            subprocess.run([sys.executable, str(ROOT / "scripts/plot_results.py"), "--out-dir", str(out)],
                           check=True, capture_output=True, text=True)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertFalse(manifest["torch_imported"])
            self.assertEqual(len(manifest["outputs"]), 14)
            for name in manifest["outputs"]:
                data = (out / name).read_bytes()
                self.assertGreater(len(data), 1000)
                self.assertTrue(data.startswith(b"%PDF") if name.endswith("pdf") else data.startswith(b"\x89PNG"))
            with (out / "checkpoint_summary.csv").open() as f:
                summaries = list(csv.DictReader(f))
            self.assertEqual(len(summaries), 138)
            matched = [r for r in summaries if (r["model"], r["checkpoint"], r["reward"])
                       == ("OLMo-3-7B", "O4", "random")][0]
            self.assertEqual(float(matched["terminal_mean_pct"]), 47.6875)

    def test_existing_raw_log_plot_cli_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ck = root / "p1_5B"
            ck.mkdir()
            records = [{"phase": "meta", "reward_mode": "random", "seed": 31001},
                       {"phase": "before", "accuracy": 0}, {"phase": "after", "accuracy": 0.005}]
            (ck / "rlvr_fixture.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            output = root / "plot.png"
            subprocess.run([sys.executable, str(ROOT / "src/checkpoint_curve/plot_three_scores.py"),
                            str(root), "--out", str(output)], check=True, capture_output=True, text=True)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
