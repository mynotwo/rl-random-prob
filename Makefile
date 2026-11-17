# GPU-free checks; CI (.github/workflows/plots.yml) runs these targets.
PYTHON ?= python
OFFLINE = HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=

.PHONY: check check-torch

# Needs only requirements-plot.txt: no torch, model weights or network.
check:
	$(PYTHON) -m compileall -q scripts tests src/checkpoint_curve/plot_three_scores.py src/cross_model/summarize.py
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) src/trainer/test_protocol.py
	$(PYTHON) src/checkpoint_scan/test_contract.py
	$(PYTHON) scripts/export_paper_results.py --check
	$(OFFLINE) $(PYTHON) scripts/plot_results.py --out-dir outputs/paper

# Needs requirements.txt (torch, transformers, peft); a 2-layer random model stands in for the 7B one.
check-torch:
	$(OFFLINE) $(PYTHON) src/shared_rollout/test_shared_rollouts.py
