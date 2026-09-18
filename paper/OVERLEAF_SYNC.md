# Overleaf synchronization contract

The repository branch `ecir-2027-redesign` is the scientific source of truth. Overleaf mirrors the anonymous contents of `paper/`; method, model, dataset, analysis, table, budget, and mitigation decisions must be changed in the repository first.

## Core manuscript sources

The upload bundle contains `main.tex`, the benchmark/model table, RQ design table, related-work table, bibliography files, and methodology PDFs. `results_contract_table.tex` is the stable paper-facing result entrypoint.

For RQ1/RQ2 and RQ3/RQ4 it prefers generated audited tables when they exist and otherwise falls back to the pre-registered TBD contracts:

- `generated/rq12_inference_table.tex` -> fallback `result_tables/rq12_main_table.tex`;
- `generated/rq34_results_table.tex` -> fallback `result_tables/rq34_main_table.tex`.

Hosted FairSynth is different: `generated/fairsynth_hosted_table.tex` is included **only if an audited hosted FairSynth inference artifact has actually been rendered**. There is no synthetic fake-number fallback.

The full-scale local white-box stratum follows the same artifact-only rule. `generated/whitebox_summary_table.tex` is included only after audited local output has been analyzed and rendered. White-box numerical values are never typed into the manuscript by hand and are never pooled with unavailable hosted-model internals.

The bundle also carries the compact supporting result contracts:

- `result_tables/coverage_table.tex`;
- `result_tables/trait_ablation_table.tex`;
- `result_tables/fairsynth_table.tex`;
- `result_tables/whitebox_table.tex`.

Generated LaTeX result tables under `paper/generated/` and generated result PDFs are automatically copied into the Overleaf ZIP when present. The bundler therefore cannot silently drop numerical artifacts that the local paper already consumes.

## Pre-execution scientific seal

Before any hosted generation or local model inference, build the zero-call seal:

```bash
python scripts/build_preexecution_seal.py \
  --output-dir results/preexecution/seal-v1
```

The seal regenerates the canonical FairSynth freeze and both deterministic FairSynth plans, renders pre-result contracts/figures, runs configuration and manuscript preflights, hashes the tracked scientific specification, records unresolved real-dataset blockers, and emits a pre-result anonymous Overleaf ZIP. It records that hosted generation calls are zero, local model weights were not loaded, and empirical results were not seen or inserted.

Real hosted and local launchers require the seal to match both the checked-out Git commit and the immutable plan SHA. Therefore changing executable scientific source after sealing requires a new seal and experiment version rather than continuing an old run log.

## Hosted API execution contract

The current black-box phase uses the Zhizengzeng OpenAI-compatible gateway and the six exact IDs frozen in `configs/models.yaml`. Paid execution is guarded by:

- live `GET /v1/models` exact-ID verification;
- a 200 RMB normal stop target;
- a 250 RMB client-side emergency stop threshold;
- a 2 RMB pre-cell reserve;
- gateway balance reconciliation before/after every persisted hosted cell;
- an exact pre-execution seal and Git-SHA match.

The 250 RMB value is deliberately **not represented as an atomic provider-side spend cap**. FairEval normally stops around 200 RMB and keeps roughly 50 RMB of operational headroom because a provider could, in principle, charge an in-flight request before the next balance reconciliation. The budget remains outcome-independent: cells execute in immutable plan order and the budget is never expanded after inspecting results. Any budget-truncated coverage is reported as incomplete planned coverage rather than evidence for or against an RQ.

The first hosted executable evidence layer is the deterministic A/B/C-balanced FairSynth subset compiled by `scripts/plan_hosted_fairsynth.py`. Real-world hosted RQ1/RQ2 remains blocked until exact third-party raw releases are frozen.

## Full-scale local white-box execution contract

The two frozen open-weight models form a **required, separately reported replication stratum** rather than a small optional demo. The model panel remains fixed to exact Hugging Face revisions in `configs/local_models.yaml`; adding models after observing results is prohibited by the campaign contract.

The canonical controlled run is:

- 360 FairSynth users;
- 6 registered conditions per user;
- 3 seeded repetitions;
- 2 frozen local model families;
- 12,960 core generations in total.

After exact real-world dataset freezes exist, the same two models replicate the executable RQ1/RQ2 core, registered RQ3 robustness factors, and RQ4 mitigation path. Hosted and local estimands remain separate strata.

Build the immutable full core plan with:

```bash
python scripts/build_whitebox_campaign.py \
  --freeze-root data/frozen \
  --output-root results/plans/whitebox-full-v1
```

When all six real-world freezes are available, rebuild/version the campaign with `--include-real-world` rather than modifying an existing plan in place.

Run one family at a time so a large GPU is used for throughput without mixing model state:

```bash
python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family qwen25_local \
  --output-jsonl results/runs/whitebox-qwen25-v1.jsonl \
  --max-cells 250
```

Review the dry run first. Real GPU execution additionally requires `--execute --code-commit-sha <SHA>` and the matching default pre-execution seal. Repeat for `phi35_local`. The output file is resume-safe and every completed plan cell is persisted before the next cell starts.

For the registered local RQ3 prompt/cue/order/cutoff/stochasticity factors, compile the separate seeded extra-cell plan only after real-world freezes exist:

```bash
python scripts/plan_whitebox_rq3.py \
  --freeze-root data/frozen \
  --output-dir results/plans/whitebox-full-v1/rq3
```

This planner reuses the same registered semantic geometry as hosted RQ3, substitutes the frozen local model panel, freezes a deterministic generation seed into every local robustness cell, and rehashes the immutable cell ID. Execute it with the same one-family `scripts/run_whitebox_family.py` runner by pointing `--plan-dir` to the RQ3 plan and using a seal/version that contains that exact plan.

RQ4 identity-irrelevance prompting is derived from the real-world local core with `scripts/build_rq4_prompt_plan.py`; because the source local cells already carry frozen generation seeds, the derived executor-compatible plan preserves those seeds while changing only the named prompt intervention. Contextual PAIR remains a post-processing mitigation and never selects an operating point on test outcomes.

## Result generation

Before experiments:

```bash
python scripts/render_result_tables.py --contracts-only
```

After a hosted FairSynth run exists, the canonical hosted paper update is:

```bash
python scripts/finalize_hosted_fairsynth.py
```

That path performs run-log audit, FairSynth scoring/inference, `generated/fairsynth_hosted_table.tex` rendering, manuscript/result preflights, Overleaf bundle rebuild, and paper-sync hashing.

For a completed local white-box run, use the full finalizer rather than manually combining families:

```bash
python scripts/finalize_whitebox.py \
  --input-jsonl results/runs/whitebox-qwen25-v1.jsonl \
  --input-jsonl results/runs/whitebox-phi35-v1.jsonl \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --output-dir results/analysis/whitebox-full-v1 \
  --paper-table paper/generated/whitebox_summary_table.tex \
  --overleaf-output dist/FairEval_ECIR2027_Overleaf.zip
```

The finalizer audits both family logs and seed contracts, refuses incomplete/duplicate/mixed coverage, merges in immutable plan order, computes the declared local diagnostics, renders `paper/generated/whitebox_summary_table.tex`, runs paper preflights, and rebuilds the anonymous Overleaf ZIP. No manual averaging is permitted.

For the complete real-world RQ1--RQ4 study, use the final renderer documented in `docs/RESULT_TABLES.md`. Final rendering requires the core inference, C5 trait inference, RQ3 summary, RQ4 PAIR artifact, RQ4 identity-irrelevance prompting summary, FairSynth inference, local white-box summary, and the user-condition artifact for every claimed stratum.

RQ4 identity-irrelevance prompting runs through its own derived immutable plan (`scripts/build_rq4_prompt_plan.py`) so the neutral RQ1 audit is never fairness-coached. The derived plan uses the same `faireval-run-plan-v1` execution/audit protocol and changes only the named prompt mode while preserving source-cell provenance.

## Figures

Conceptual/non-result figures are generated from `scripts/build_paper_figures.py` as matched publication PDFs and editable SVG companions. The current conceptual set is `faireval_framework`, `faireval_conditions`, and `faireval_evaluation_pipeline`; the SVGs preserve text as text and are bundled with the PDFs for Overleaf handoff. Figure 1 is the motivating matched-context example, Figure 2 summarizes the registered RQ geometry, and Figure 3 documents the artifact-to-claim audit path. Result PDFs are generated only from frozen analysis artifacts by `scripts/build_result_figures.py`; until then `main.tex` renders explicit placeholders. Styling changes to empirical result figures must be made in the renderer and regenerated from the same audited artifact rather than by editing plotted values or geometry manually.

## Double blind and integrity

- keep anonymous author/institution metadata until camera-ready;
- do not insert numerical findings manually;
- never describe FairSynth labels as human demographics or its OCEAN vectors as measured psychometrics;
- never describe gateway-native thinking/sampling controls as verified unless returned metadata proves them;
- never describe local token-score diagnostics as calibrated uncertainty;
- never pool local internal-score diagnostics with hosted models that do not expose the same internals;
- never select a PAIR point from test outcomes or tune one per model/dataset;
- never use the RQ4 mitigation prompt in the RQ1 audit run;
- never increase the hosted budget after inspecting results;
- never describe the 250 RMB client-side threshold as a provider-enforced atomic spending cap;
- never add a white-box model after observing campaign outcomes without versioning a new study;
- never continue execution after the code/plan no longer matches the pre-execution seal.

Before an Overleaf upload, run the repository CI/preflight chain. CI checks configuration, manuscript sources, methodology figures, result-table contracts, the zero-call seal path, and builds an anonymous Overleaf ZIP only after the branch passes its guards.
