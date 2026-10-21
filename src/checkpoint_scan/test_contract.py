import json
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).parent


class ContractTest(unittest.TestCase):
    def test_manifest_has_required_24_points(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(len(manifest["checkpoints"]), 24)
        self.assertEqual(len({x["commit"] for x in manifest["checkpoints"]}), 24)

    def test_worker_has_complete_training_contract(self):
        source = (ROOT / "worker.py").read_text()
        spec = importlib.util.spec_from_file_location("scan24_worker", ROOT / "worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contract = module.build_contract()
        random_seeds = [item[1][1] for item in contract if item[1] and item[1][0] == "random"]
        gt_seeds = [item[1][1] for item in contract if item[1] and item[1][0] == "ground_truth"]
        self.assertEqual(random_seeds, list(range(31001, 31009)))
        self.assertEqual(gt_seeds, [31001, 31002])
        self.assertIn('"--save_adapter"', source)
        self.assertIn('"--sample_seed_start", "41001"', source)
        self.assertIn('if remaining <= 2700', source)


if __name__ == "__main__":
    unittest.main()
