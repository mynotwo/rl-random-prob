"""Golden outputs of the torch-free job generators: job lists, manifests, Slurm scripts and stdout.

After an intended change, regenerate with
    UPDATE_GOLDEN=1 python -m unittest discover -s tests
and review the diff of tests/golden/gen_jobs/.

cross_model's tokenizer check needs transformers and the Hugging Face hub, so a stub replaces it here.
shared_rollout/gen_jobs.py imports torch and is not covered.
"""
import contextlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GOLDEN = ROOT / "tests/golden/gen_jobs"
# PROBE_PYTHON is written into every command; PROBE_OUT is read by cross_model.
ENV = {k: v for k, v in os.environ.items() if k not in ("PROBE_PYTHON", "PROBE_OUT")}


def run_script(script, args):
    """Launch a generator the way the README does: from src/, by its relative path."""
    return subprocess.run([sys.executable, script, *args], cwd=SRC, env=ENV, check=True,
                          capture_output=True, text=True).stdout


def run_cross_model(args):
    with mock.patch.dict(os.environ, ENV, clear=True), mock.patch.object(sys, "path", list(sys.path)), \
            mock.patch.object(sys, "argv", ["gen_jobs.py", *args]), contextlib.redirect_stdout(io.StringIO()) as out:
        spec = importlib.util.spec_from_file_location("cross_model_gen_jobs", SRC / "cross_model/gen_jobs.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        generator.prompt_check = lambda ckpts: ({c[0]: {"stub": True} for c in ckpts}, "<eval_ids_sha256>")
        generator.main()
    return out.getvalue()


class GenJobsGoldenTest(unittest.TestCase):
    maxDiff = 4000

    def check(self, case, run, invocations):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            stdout = "".join(run(args + ["--out", str(out)]) for args in invocations)
            files = {"stdout.txt": stdout}
            files.update((p.relative_to(out).as_posix(), p.read_text()) for p in sorted(out.rglob("*")) if p.is_file())
            for name, text in files.items():
                text = text.replace(str(out), "<OUT>").replace(str(ROOT), "<REPO>")
                files[name] = re.sub(r'("planned(?:_utc)?": ")[^"]+', r"\1<UTC>", text)
        golden = GOLDEN / case
        if os.environ.get("UPDATE_GOLDEN"):
            shutil.rmtree(golden, ignore_errors=True)
            golden.mkdir(parents=True)
            for name, text in files.items():
                (golden / name).write_text(text)
        self.assertEqual(sorted(files), sorted(p.name for p in golden.iterdir()))
        for name, text in files.items():
            self.assertEqual(text, (golden / name).read_text(), f"{case}/{name}")

    def test_checkpoint_curve(self):
        self.check("checkpoint_curve", lambda args: run_script("checkpoint_curve/gen_jobs.py", args), [[]])

    def test_input_controls_one_checkpoint(self):
        self.check("input_controls_one_checkpoint", lambda args: run_script("input_controls/gen_jobs.py", args),
                   [["--tag", "p2_50B", "--arms", "C", "A", "--seeds", "31001", "31002"]])

    def test_input_controls_other_models_and_raw_prompt_arm(self):
        self.check("input_controls_other_models", lambda args: run_script("input_controls/gen_jobs.py", args),
                   [["--tag", "qwen25_7b", "llama31_8b", "--arms", "D", "E", "--seeds", "31001", "31002"]])

    def test_cross_model_phases(self):
        self.check("cross_model", run_cross_model, [["--phase", "frozen"], ["--phase", "smoke"], ["--phase", "rl"]])

    def test_cross_model_finished_base_models(self):
        self.check("cross_model_finished_models", run_cross_model, [["--set", "shao_models_raw_20260920"]])


if __name__ == "__main__":
    unittest.main()
