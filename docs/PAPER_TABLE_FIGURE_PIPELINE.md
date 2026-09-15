# FairEval paper table + figure pipeline

The ECIR manuscript is designed so paper-facing numbers and result figures can be regenerated from frozen artifacts. Do not manually type final empirical values into LaTeX.

## Static design assets

These are pre-run scientific-design artifacts and may be committed before experiments:

- `paper/related_work_table.tex` — evidence-audited literature design coverage.
- `paper/benchmark_model_table.tex` — one combined six-dataset + six-LLM/API panel.
- `paper/rq_design_table.tex` — RQ/contrast/estimand/inference/robustness contract.
- `paper/results_contract_table.tex` — multi-panel RQ1–RQ4 placeholder contract; numerical cells remain `TBD` until generated evidence exists.
- `paper/figures/faireval_framework.pdf`
- `paper/figures/faireval_conditions.pdf`
- `paper/figures/faireval_evaluation_pipeline.pdf`

## Generated empirical artifacts

After the frozen run is complete:

```bash
python scripts/analyze_core.py \
  --output-jsonl results/raw/core-v1.jsonl \
  --plan-dir results/plans/core-v1 \
  --freeze-root data/frozen \
  --output-dir results/analysis/core-v1
```

This writes the audited RQ1/RQ2 pairs and `inference.jsonl`.

### Dataset-level RQ1/RQ2 LaTeX table

```bash
python scripts/render_paper_results.py \
  --inference-jsonl results/analysis/core-v1/inference.jsonl \
  --output paper/generated/rq12_inference_table.tex
```

The generated table reports dataset × model-family paired estimates, bootstrap confidence intervals, Holm-adjusted paired-permutation p-values, and matched rank-biserial effects. It contains no manually entered result values.

### RQ1 consequence-vs-sensitivity plot

```bash
python scripts/build_result_figures.py \
  --rq1-pairs results/analysis/core-v1/rq1_pairs.jsonl
```

This generates `paper/figures/rq1_quadrant.pdf` from paired user-level artifacts. X is `1 - RBO@K` for repetitions where both conditions yielded valid rankings. Y is the signed counterfactual nDCG gap. Invalid runs are **not** assigned an invented RBO value; their consequence is carried through primary zero-utility analysis and IOD.

### RQ2 personality-value forest plot

```bash
python scripts/build_result_figures.py \
  --inference results/analysis/core-v1/inference.jsonl
```

This generates `paper/figures/rq2_personality_forest.pdf` from RQ2 true-vs-shuffled nDCG summaries and paired bootstrap confidence intervals.

## RQ3 and RQ4 are intentionally guarded

`paper/main.tex` already reserves:

- `rq3_variance.pdf`
- `rq4_pareto.pdf`

but `scripts/build_result_figures.py` does **not** invent their input schema. If `--rq3-artifact` or `--rq4-artifact` is supplied before those analysis contracts are frozen, the generator fails deliberately.

The correct order is:

1. implement and test RQ3/RQ4 analysis rows;
2. freeze their JSONL schemas;
3. add regression fixtures;
4. only then implement the visual renderer;
5. render PDF;
6. inspect a 200-DPI rasterization for clipping/overlap/glyph failures;
7. allow the manuscript to include the result figure.

## Page-budget rule

ECIR allows 12 pages plus references. The development branch intentionally contains richer tables than the final camera-facing 12-page selection. Once real result densities are known, choose the smallest non-redundant set:

1. keep the combined benchmark/model table;
2. keep either the full RQ design table or a compressed methods paragraph if page pressure is severe;
3. keep the related-work table only if it materially clarifies joint novelty;
4. prefer dataset-level generated evidence over a second hand-written model-family table;
5. move no essential methodological qualification into an appendix because appendices count toward the 12-page limit.

## Integrity checks

Before export:

```bash
python scripts/check_paper_source.py
python scripts/check_config_consistency.py
pytest -q
```

Final empirical figures must be rendered from frozen artifacts and visually inspected. No chart should be produced from synthetic placeholder values merely to make the draft look finished.
