# FairEval — ECIR 2027 Reboot

**Target submission title:** *FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness*

This repository is a clean-room scientific redesign of FairEval for the **ECIR 2027 Full Paper Track**. It is not a cosmetic extension of the earlier paper: the evaluation target, datasets, prompt controls, statistical design, reliability analysis, and mitigation study are all rebuilt around the central distinction between useful personalization and harmful identity-conditioned disparity.

## Core scientific change

The earlier setup could treat recommendation-list change after adding identity/personality information as evidence of unfairness. The ECIR study instead separates four quantities:

1. **Behavioral shift** — did the ranking change?
2. **Held-out utility consequence** — did the change improve, preserve, or harm recommendation quality?
3. **Counterfactual demographic effect** — does demographic context change utility/exposure when preference evidence and candidates are held fixed?
4. **Grounded personality value** — does a measured Big Five profile add user-specific recommendation value beyond observed preferences and a shuffled-profile negative control?

A changed list is therefore **not automatically labeled unfair**. List similarity remains a sensitivity diagnostic; fairness claims are tied to consequence-aware paired utility/exposure evidence and clearly defined group analyses.

## Four research questions

- **RQ1 — Consequence of demographic context.** Holding observed preferences, candidates, task, prompt structure, and generation settings fixed, when demographic context changes a ranking, does the change improve utility, leave utility effectively unchanged, or create a harmful counterfactual utility/exposure disparity?
- **RQ2 — Grounded personality value.** Does measured Big Five personality provide user-specific recommendation value beyond observed preference history, and does that value survive shuffled-profile and one-trait counterfactual controls?
- **RQ3 — Reliability and generalization.** How stable are FairEval conclusions across six datasets, six LLM families, prompt/cue formulations, candidate order, ranking cutoffs, and repeated generations, and which factors explain the largest variation?
- **RQ4 — Personalization-preserving mitigation.** Can instruction-based and counterfactual re-ranking interventions reduce harmful identity-conditioned gaps while preserving overall recommendation utility and beneficial personality-driven personalization?

## Planned benchmark

### Personality-grounded track

- **GroupLens Personality 2018** — movies + measured Big Five
- **Music Master / BFI-2** — music + five BFI-2 domains, 15 facets, three rating types, and audio features
- **REASONER** — short video + CBF-PI-15 Big Five responses

### Demographic/generalization track

- **MovieLens-1M** — movie interactions + dataset-supplied demographic fields
- **Last.fm-1K** (or a pre-frozen compatible Last.fm release) — music histories + profile fields where available/licensed
- **MIND** — news click histories + logged impression candidates; no observed-demographic fairness claim is made from MIND

Every adapter records which fields are observed, derived, or synthetically counterfactualized. Protected attributes are never inferred from names, text, or embeddings.

## Six LLM families

The pre-run model manifest uses one frozen representative from each family:

- OpenAI — `gpt-5.6-terra`, reasoning `none`
- Anthropic — `claude-sonnet-5`, thinking disabled
- Google — `gemini-3.8-flash`, thinking `low`
- DeepSeek — V4.1 Flash through API model `deepseek-flash`, thinking disabled
- Alibaba Qwen — `qwen3.8-max-0902`, thinking disabled
- Meta — `meta-llama/Llama-4-Maverick-17B-128E-Instruct`; serving provider/revision still must be frozen before pilot

Exact callable IDs, provider revisions/regions, request timestamps, requested and applied decoding controls, prompt hashes, response hashes, and code commit SHA are stored in run manifests. Moving provider aliases must not silently substitute a different backend during the main experiment. FairEval requests low-variance sampling where supported but does **not** claim exact cross-provider decoding equivalence.

## Prompt design

Prompt design is a first-class experimental variable rather than hidden wording:

- a **neutral audit system prompt** for RQ1--RQ3 with no fairness coaching;
- a separately named **identity-irrelevance mitigation prompt** for RQ4;
- three pre-registered task paraphrases;
- multiple explicit demographic cue representations;
- fixed top-level fields with `unspecified` null conditions;
- psychometric instrument metadata accompanying normalized OCEAN values;
- deterministic candidate-order seeds reused inside paired counterfactuals;
- exact-K candidate-ID JSON output with no chain-of-thought or rationale request;
- prompt SHA-256 logged for every run.

Executable tests verify that a pure demographic counterfactual changes only the demographic field and a pure personality counterfactual changes only the personality field.

## Factorized experiment design

The study deliberately avoids a wasteful full Cartesian product. `configs/study_design.yaml` defines:

- **RQ1:** observed-demographic datasets and paired demographic interventions;
- **RQ2:** the three measured-personality datasets with true, shuffled, and one-trait controls;
- **RQ3:** pre-registered prompt/cue/order/stochasticity robustness subsets across all six families/datasets;
- **RQ4:** demographic datasets with instruction baselines and PAIR re-ranking;
- **MIND:** candidate-constrained domain/reliability evaluation, with any identity condition labeled only as a synthetic stress test.

The sample-size plan uses a variance/invalidity pilot, then freezes confirmatory N without using the pilot treatment-effect mean to chase significance.

## Primary metrics

**Recommendation utility:** nDCG@10, Recall@10; MRR as secondary.

**Counterfactual fairness/consequence:** signed and absolute Counterfactual Utility Gap (CUG), Group Utility Disparity when justified, Counterfactual Exposure Gap only when complete auditable top-K group metadata exists, and Invalid Output Disparity.

**Personality:** Personality Value Added (PVA), comparing the true measured profile with a deterministic whole-profile derangement, plus pre-registered one-trait observed-value interventions.

**Sensitivity only:** RBO/Jaccard and the old PAFS-style quantity are diagnostics, not primary fairness definitions.

## Mitigation

RQ4 compares the unmitigated audit condition with:

1. identity-irrelevance instruction;
2. a counterfactually fair prompting baseline;
3. **PAIR — Preference-Aligned Identity Re-ranking**, whose trade-off parameter is chosen on validation data only.

Success requires reducing harmful utility/exposure gaps **without merely forcing rankings to look similar**. Utility--fairness Pareto frontiers are reported instead of selecting a post-hoc operating point.

## Repository layout

```text
configs/          dataset/model/experiment/study manifests
scripts/          environment, manuscript, and vector-figure preflight/build tools
src/faireval/     benchmark, adapters, providers, metrics, prompts, runner, mitigation
tests/            metric, prompt-invariant, runner, adapter, provider, and statistics tests
docs/             blueprint, literature audit, reviewer traceability, novelty and submission guards
paper/            anonymous LNCS source, audited comparison table, and vector PDF figures
results/          generated outputs only; large/raw/sensitive outputs are not committed
```

## Reproducible setup

Python **3.10+** is required; CI currently exercises Python 3.11.

### Option A — standard requirements file

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins provider SDK versions because provider request/response schemas are part of the experimental apparatus. Core scientific packages use bounded compatible ranges. At every pilot/confirmatory freeze, save the exact environment with `python -m pip freeze` alongside the run manifest.

### Option B — editable research install

```bash
python -m pip install -e '.[analysis,providers,dev]'
```

### Provider credentials

```bash
cp .env.example .env
```

Never commit `.env` or real API keys. Qwen requires a region-compatible `QWEN_BASE_URL`; Llama requires a frozen provider name/base URL before the pilot. `scripts/check_environment.py` validates the preflight state without making model calls.

```bash
python scripts/check_environment.py
```

## Frozen-data and execution workflow

The CLI is intentionally dry-run first.

```bash
faireval dataset-card --dataset movielens_1m
faireval prepare --dataset movielens_1m --raw-dir <RAW> --output-dir frozen/movielens_1m
faireval verify-freeze --output-dir frozen/movielens_1m

# After all six datasets are frozen:
faireval plan-core --freeze-root frozen --output-dir plans/core

# Inspect pending cells: zero API calls by default.
faireval execute-plan \
  --plan-dir plans/core \
  --freeze-root frozen \
  --output-jsonl results/core.jsonl

# Actual execution requires explicit opt-in and the exact code commit SHA.
faireval execute-plan \
  --plan-dir plans/core \
  --freeze-root frozen \
  --output-jsonl results/core.jsonl \
  --code-commit-sha <GIT_SHA> \
  --execute
```

Invalid model responses count as experimental outcomes after persistence; execution does not silently retry a cell until it becomes valid.

## Paper and vector figures

All framework assets are regenerated from source rather than edited manually:

```bash
python scripts/build_paper_figures.py
python scripts/check_paper_source.py
```

Generated vector PDFs:

```text
paper/figures/faireval_framework.pdf
paper/figures/faireval_conditions.pdf
paper/figures/faireval_evaluation_pipeline.pdf
```

The manuscript imports `paper/related_work_table.tex`, whose 16-row comparison matrix is backed by `docs/LITERATURE_COMPARISON_AUDIT.md`. A red check means explicitly documented coverage, a gray circle means partial/adjacent coverage, and a black cross means the feature was not established by the audited evidence; the symbols are **not** claims of empirical superiority.

The repository intentionally does not vendor an unofficial/modified Springer LNCS class. See `paper/README.md` for the official-template compile step.

## Integrity and reproducibility rules

- No empirical number is invented or hand-entered into the final paper.
- Pilot and confirmatory outputs are labeled and separated.
- Invalid generations are retained as observable behavior.
- No prompt/model/dataset/K/hyperparameter is selected after viewing test outcomes and then presented as confirmatory.
- Tables and figures are generated from frozen result artifacts and record input hashes + analysis commit.
- Existing FairEval/author-publication overlap is documented and handled under ECIR double-blind rules.
- Literature-comparison symbols are backed by an evidence ledger; ambiguous coverage stays partial rather than being upgraded to make FairEval look stronger.

## Continuous integration

The GitHub Actions workflow now checks:

- unit tests;
- critical Python lint failures;
- deterministic regeneration of the three vector PDF figures;
- PDF signatures/minimum-size sanity checks;
- missing BibTeX citation keys;
- missing LaTeX inputs/manuscript assets;
- stale model-panel wording;
- double-blind author placeholder; and
- the result-integrity guard in the abstract.

## ECIR constraints

- Full-paper abstract registration: **21 September 2026**.
- Full-paper deadline: **5 October 2026**.
- Springer LNCS format.
- Maximum **12 pages plus references**; appendices count inside the 12-page limit.
- Double-blind review.

See `docs/ECIR2027_SUBMISSION_COMPLIANCE.md` before export.

## Current status

- scientific redesign: implemented
- four RQs + metrics + mitigation formulation: implemented
- six dataset configurations/adapters: implemented, including Music Master/BFI-2
- six model-family manifest/provider layer: implemented; final Llama hosting snapshot still must be frozen before pilot
- controlled prompt/cue/order machinery: implemented + unit-tested
- immutable run planning/execution + resume guards: implemented
- standard `requirements.txt` + environment preflight: implemented
- academic framework/condition/evaluation vector PDFs + regeneration script: implemented
- 15-prior-work + FairEval comparison matrix with source audit: implemented
- manuscript citation/input/asset preflight: implemented
- result values: intentionally **not populated** until experiments run
- next execution milestone: validate raw dataset schemas/checksums, run the variance/invalidity pilot, freeze confirmatory N, then execute the locked matrix
