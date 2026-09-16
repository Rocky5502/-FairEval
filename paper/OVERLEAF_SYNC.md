# Overleaf synchronization contract

The repository branch `ecir-2027-redesign` is the scientific source of truth. Overleaf mirrors the anonymous contents of `paper/`; method, model, dataset, analysis, table, and mitigation decisions must be changed in the repository first.

## Core manuscript sources

The upload bundle contains `main.tex`, the benchmark/model table, RQ design table, related-work table, bibliography files, and the methodology PDFs. `results_contract_table.tex` is a stable entrypoint that inputs two main result tables:

- `result_tables/rq12_main_table.tex` — RQ1/RQ2 primary effects;
- `result_tables/rq34_main_table.tex` — RQ3 reliability + three-method RQ4 mitigation.

The bundle also always carries four compact supporting result tables:

- `result_tables/coverage_table.tex`;
- `result_tables/trait_ablation_table.tex`;
- `result_tables/fairsynth_table.tex`;
- `result_tables/whitebox_table.tex`.

These compact tables are evidence-ready but are not automatically inserted into the 12-page body. Promotion is a later page-budget/story decision, not a reason to alter their analysis code.

## Result generation

Before experiments:

```bash
python scripts/render_result_tables.py --contracts-only
```

After frozen artifacts exist, use the final renderer documented in `docs/RESULT_TABLES.md`. Final rendering requires the core inference, C5 trait inference, RQ3 summary, RQ4 PAIR artifact, RQ4 identity-irrelevance prompting summary, FairSynth inference, local white-box summary, and user-condition artifact. A missing artifact is a hard error.

RQ4 identity-irrelevance prompting runs through its own derived immutable plan (`scripts/build_rq4_prompt_plan.py`) so the neutral RQ1 audit is never fairness-coached. The derived plan uses the same `faireval-run-plan-v1` execution/audit protocol and changes only the named prompt mode while preserving source-cell provenance.

## Figures

Methodology PDFs are generated from `scripts/build_paper_figures.py`. Result PDFs are generated only from frozen analysis artifacts by `scripts/build_result_figures.py`; until then `main.tex` renders explicit placeholders.

## Double blind and integrity

- keep anonymous author/institution metadata until camera-ready;
- do not insert numerical findings manually;
- never describe FairSynth labels as human demographics or its OCEAN vectors as measured psychometrics;
- never describe local token-score diagnostics as calibrated uncertainty;
- never select a PAIR point from test outcomes or tune one per model/dataset;
- never use the RQ4 mitigation prompt in the RQ1 audit run.

Before an Overleaf upload, run the repository CI/preflight chain. CI checks configuration, manuscript sources, methodology figures, the six-table contract, and builds an anonymous Overleaf ZIP only after the branch passes its guards.
