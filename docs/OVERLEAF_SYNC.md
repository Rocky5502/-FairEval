# FairEval ECIR 2027 — Overleaf Synchronization Contract

The `paper/` directory on branch `ecir-2027-redesign` is the canonical LaTeX source for the ECIR 2027 manuscript. The Overleaf project should mirror this directory rather than maintaining independent edits.

## Files that must be synchronized

Upload/update these text sources together:

- `paper/main.tex`
- `paper/benchmark_model_table.tex`
- `paper/rq_design_table.tex`
- `paper/results_contract_table.tex`
- `paper/related_work_table.tex`
- `paper/references.bib`
- `paper/references_extra.bib`

The manuscript currently represents the following frozen design:

- six real-world datasets for real-world claims;
- FairSynth-360 as a separately reported synthetic sanity benchmark;
- six hosted/API model families;
- two local open-weight models (`Qwen/Qwen2.5-7B-Instruct` and `microsoft/Phi-3.5-mini-instruct`);
- local-only token-score diagnostics labeled exploratory and uncalibrated;
- contextual PAIR with validation-only operating-point selection;
- one global PAIR point chosen on a deterministic 20% validation split (seed 2027), retaining at least 95% of unmitigated validation nDCG before minimizing absolute CUG;
- no test-set or per-model/per-dataset operating-point tuning.

## Methodology figures

The canonical editable source is:

```text
scripts/build_paper_figures.py
```

It regenerates:

```text
paper/figures/faireval_framework.pdf
paper/figures/faireval_conditions.pdf
paper/figures/faireval_evaluation_pipeline.pdf
```

The source code, not a manually edited PDF, defines the figure content. If the PDFs are regenerated locally, replace all three Overleaf assets in the same synchronization step so captions, manuscript text, and figures cannot drift.

The current framework figure explicitly shows:

- six real-world datasets plus FairSynth-360;
- six hosted/API plus two local open-weight model configurations;
- C0--C5 controlled conditions;
- CUG and PVA consequence estimands;
- local white-box diagnostic branch;
- contextual PAIR with the validation-only >=95% utility-retention rule.

## Result figures

Result figures are evidence-generated artifacts, not manually drawn illustrations. Before experiments, missing result figures intentionally compile as placeholders. After analysis, generate them from frozen analysis JSON/JSONL artifacts. Never enter a numerical result manually solely to improve the paper presentation.

Expected result assets:

```text
paper/figures/rq1_quadrant.pdf
paper/figures/rq2_personality_forest.pdf
paper/figures/rq3_variance.pdf
paper/figures/rq4_pareto.pdf
```

## Pre-upload checks

From the repository root, run:

```bash
python scripts/check_config_consistency.py
python scripts/build_paper_figures.py
python scripts/check_paper_assets.py
python scripts/check_paper_source.py
pytest -q
```

Only synchronize to Overleaf after these checks pass. CI executes the same manuscript/config guards.

## Double-blind guard

Keep:

```tex
\author{Anonymous Authors}
\authorrunning{Anonymous Authors}
\institute{Anonymous Institution}
```

until the review policy permits de-anonymization. Do not put author-identifying repository URLs, acknowledgments, or self-referential phrasing into the anonymous manuscript.

## Integrity rule

The repository is the source of truth. If an Overleaf edit changes a method, metric, dataset/model scope, hyperparameter grid, table definition, citation, or result, port that edit back to this branch immediately and let CI validate it. Do not allow an Overleaf-only scientific definition to diverge from executable code/configuration.
