"""The three copies of the on-policy trainer, compared as text.

They need torch and model weights, so they cannot run here. The reference copy is pinned by hash in
configs/olmo2_checkpoints.json and every job manifest records the hash of the copy it launches, so the copies are
not merged; this test fails as soon as one of them drifts from the others.
"""
import ast
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "src/trainer/rlvr_probe_v2_fixedsplit.py"
RAW_PROMPT = ROOT / "src/trainer/rlvr_probe_v2_fixedsplit_rawprompt.py"
EARLIER = ROOT / "src/rlvr_probe_v2.py"
BUILD_PROMPT = "def build_prompt(ex: Dict[str, str], mode: str) -> str:\n"
RAW_PROMPT_BRANCH = '''\
    # random-prompt control (2026-09-20): a row that carries `raw_prompt` IS the whole prompt, no template.
    # Rows without the field (every GSM row, so the whole eval set) take the original path unchanged.
    if ex.get("raw_prompt") is not None:
        return ex["raw_prompt"]
'''


def functions(path):
    """{name: source text including decorators} of the module-level functions, in file order."""
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    return {node.name: "".join(lines[min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1:node.end_lineno])
            for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


class TrainerCopiesTest(unittest.TestCase):
    def test_reference_trainer_is_the_pinned_file(self):
        pinned = json.loads((ROOT / "configs/olmo2_checkpoints.json").read_text())["training_script_sha256_reference"]
        self.assertEqual(hashlib.sha256(REFERENCE.read_bytes()).hexdigest(), pinned)

    def test_raw_prompt_copy_adds_one_branch_to_build_prompt(self):
        reference = REFERENCE.read_text()
        self.assertEqual(reference.count(BUILD_PROMPT), 1)
        self.assertEqual(RAW_PROMPT.read_text(), reference.replace(BUILD_PROMPT, BUILD_PROMPT + RAW_PROMPT_BRANCH))

    def test_earlier_trainer_differs_only_in_main(self):
        reference, earlier = functions(REFERENCE), functions(EARLIER)
        self.assertEqual(list(earlier), list(reference))
        self.assertEqual([name for name in reference if earlier[name] != reference[name]], ["main"])


if __name__ == "__main__":
    unittest.main()
