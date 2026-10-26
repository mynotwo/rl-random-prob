# Published result tables

The CSVs are a version-pinned transcription of numerical tables in
[arXiv:2610.01066v1](https://arxiv.org/html/2610.01066v1).
They contain reported outcomes, not recovered raw logs or new model runs.
`paper_tables.json` preserves the source table cells, captions, table numbers and source URL.
`scripts/export_paper_results.py` deterministically exports them and checks column order, sample counts,
row maxima, duplicate identities, reused scores and answer-format coverage.

| CSV | Rows | Source tables | Meaning |
|---|---:|---|---|
| [checkpoint_runs.csv](checkpoint_runs.csv) | 716 | 8–13, 15 | Per-seed terminal scores, step 500: 444 GSM, 72 code, 120 lookup, 80 additional-model runs |
| [frozen_scores.csv](frozen_scores.csv) | 69 | 8, 10, 12, 14 | Published Base/Raw scores for 37 + 12 + 12 + 8 checkpoints |
| [input_controls.csv](input_controls.csv) | 160 | 18, 20 | 128 OLMo-2 and 32 Qwen2.5/Llama terminal scores |
| [shared_rollouts.csv](shared_rollouts.csv) | 64 | 23 | 16 paired replicate blocks × four response-source/learner cells |
| [answer_format.csv](answer_format.csv) | 15 | 7 | Selected GSM Random runs ending at ≥50% lenient accuracy, with strict scores and format rates |

There are 940 terminal appearances across the first, third and fourth files. The eight B-arm seeds
31001–31008 in `input_controls.csv` reuse the GSM Random P2 +50B results. Their `reuses_run_id` points
to the original `checkpoint_runs.csv` identity. Resolve that link before counting distinct runs:
**940 − 8 = 932**, as in the paper. The CSVs are not 940 independent training runs.

## Fields and units

- `accuracy_pct`, `lenient_accuracy_pct`, `strict_accuracy_pct`, `format_rate_pct`: percentages in [0, 100].
  `0.5` means 0.5%, not 50%. Plotting adapters convert to fractions only for the existing line-plot helpers.
- `step`: 500, the terminal evaluation. Intermediate scheduled peaks and complete trajectories are not included.
- `seed`: the actual training seed, e.g. 31001. The source's shortened seed headings are expanded according
  to its captions. `replicate` is the shared-rollout block, 51001–51016, preserving all four paired cells.
- `checkpoint`: OLMo-2 manifest tag, e.g. `p1_462B` or `p2_5B`, or the additional-model ID S1–S4/O1–O4.
  P2 token counts are additional mid-training tokens; the x-axis follows checkpoint order, not token scale.
- `reward`: `ground_truth` or `random`. In shared rollouts every arm uses cached random rewards;
  `response_source` and `learner` are separate fields and must not be swapped.
- `arm`: B original GSM, C word-shuffled GSM, A unrelated MMLU stems, D random tokens in the math template,
  E random tokens without that template. OLMo-2 uses P2 +50B; Qwen2.5 and Llama use the paper's base models.
- `run_id`: unique appearance identity; `reuses_run_id` is empty for an original run.
- `source_table`, `source_url`: the version-pinned paper locator for each row.

The Base/Raw values follow the paper's frozen-score convention. They are not per-run step-0 logs;
subtracting them does not reconstruct all reported own-initial-score gains. Missing/dashed seed cells
are omitted rather than imputed with zero. Do not infer per-seed predictions or evaluation-item IDs
from these rounded, aggregate scores.

## Redraw and check

From the repository root, install `requirements-plot.txt`, then:

```bash
python scripts/export_paper_results.py --check
python scripts/plot_results.py --out-dir outputs/paper
```

The plotter checks the input CSVs before drawing, without downloading anything or replacing a missing
file. `--results-dir PATH` checks a copy of this published snapshot; it is not a generic custom-run parser.
Use the existing raw-log plot/summarize scripts for your own training runs.

To regenerate CSVs from the preserved source cells:

```bash
python scripts/export_paper_results.py
```

Review any diff against the cited paper table. `--check` exits nonzero for a missing, duplicated, altered
or reordered row. The exporter never fills unrun configurations. Figure summaries report actual sample
counts and terminal maxima/means/medians; a maximum depends on seed budget and is not a capability ceiling.
The plots include the answer-format diagnostic because lenient numerical accuracy and compliance with
an answer-line format can diverge.

This snapshot covers the main 932-run experiments and frozen/format diagnostics. It does not contain
raw training logs, complete trajectories, all statistical tables, the earlier 55-run group-size/steering
study or masked-SFT results. Redrawing the reported values does not independently reproduce training
or establish the paper's causal interpretations.
