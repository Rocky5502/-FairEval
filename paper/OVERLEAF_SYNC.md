# Overleaf synchronization contract

The repository branch `ecir-2027-redesign` is the scientific source of truth. Overleaf mirrors the anonymous contents of `paper/`; method, model, dataset, analysis, table, budget, and mitigation decisions must be changed in the repository first.

## Core manuscript sources

The upload bundle contains `main.tex`, the benchmark/model table, RQ design table, related-work table, bibliography files, and methodology PDFs. `results_contract_table.tex` is the stable paper-facing result entrypoint.

For RQ1/RQ2 and RQ3/RQ4 it prefers generated audited tables when they exist and otherwise falls back to the pre-registered TBD contracts:

- `generated/rq12_inference_table.tex` -> fallback `result_tables/rq12_main_table.tex`;
- `generated/rq34_results_table.tex` -> fallback `result_tables/rq34_main_table.tex`.

Hosted FairSynth is different: `generated/fairsynth_hosted_table.tex` is included **only if an audited hosted FairSynth inference artifact has actually been rendered**. There is no synthetic fake-number fallback.

The bundle also carries the compact supporting result contracts:

- `result_tables/coverage_table.tex`;
- `result_tables/trait_ablation_table.tex`;
- `result_tables/fairsynth_table.tex`;
- `result_tables/whitebox_table.tex`.

Generated LaTeX result tables under `paper/generated/` and generated result PDFs are automatically copied into the Overleaf ZIP when present. The bundler therefore cannot silently drop numerical artifacts that the local paper already consumes.

## Hosted API execution contract

The current black-box phase uses the Zhizengzeng OpenAI-compatible gateway and the six exact IDs frozen in `configs/models.yaml`. Paid execution is guarded by:

- live `GET /v1/models` exact-ID verification;
- a 200 RMB normal stop target;
- a 250 RMB emergency hard ceiling;
- a 2 RMB pre-cell reserve;
- gateway balance reconciliation before/after every persisted hosted cell.

The budget is outcome-independent. Cells execute in immutable plan order and the budget is never expanded after inspecting results. Any budget-truncated coverage is reported as incomplete planned coverage rather than evidence for or against an RQ.

The first hosted-only executable evidence layer is the deterministic A/B/C-balanced FairSynth subset compiled by `scripts/plan_hosted_fairsynth.py`. Real-world RQ1/RQ2 remains blocked until exact third-party raw releases are frozen.

## Result generation

Before experiments:

```bash
python scripts/render_result_tables.py --contracts-only
```

After a hosted FairSynth run exists, the canonical paper update is:

```bash
python scripts/finalize_hosted_fairsynth.py
```

That path performs run-log audit, FairSynth scoring/inference, `generated/fairsynth_hosted_table.tex` rendering, manuscript/result preflights, Overleaf bundle rebuild, and paper-sync hashing.

For the complete real-world RQ1--RQ4 study, use the final renderer documented in `docs/RESULT_TABLES.md`. Final rendering requires the core inference, C5 trait inference, RQ3 summary, RQ4 PAIR artifact, RQ4 identity-irrelevance prompting summary, FairSynth inference, optional local white-box summary when that track is executed, and the user-condition artifact.

RQ4 identity-irrelevance prompting runs through its own derived immutable plan (`scripts/build_rq4_prompt_plan.py`) so the neutral RQ1 audit is never fairness-coached. The derived plan uses the same `faireval-run-plan-v1` execution/audit protocol and changes only the named prompt mode while preserving source-cell provenance.

## Figures

Methodology PDFs are generated from `scripts/build_paper_figures.py`. Result PDFs are generated only from frozen analysis artifacts by `scripts/build_result_figures.py`; until then `main.tex` renders explicit placeholders.

## Double blind and integrity

- keep anonymous author/institution metadata until camera-ready;
- do not insert numerical findings manually;
- never describe FairSynth labels as human demographics or its OCEAN vectors as measured psychometrics;
- never describe gateway-native thinking/sampling controls as verified unless returned metadata proves them;
- never describe local token-score diagnostics as calibrated uncertainty;
- never select a PAIR point from test outcomes or tune one per model/dataset;
- never use the RQ4 mitigation prompt in the RQ1 audit run;
- never increase the hosted budget after inspecting results.

Before an Overleaf upload, run the repository CI/preflight chain. CI checks configuration, manuscript sources, methodology figures, result-table contracts, and builds an anonymous Overleaf ZIP only after the branch passes its guards.
