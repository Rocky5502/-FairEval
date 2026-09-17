# FairEval

> **ECIR 2027 redesign branch:** `ecir-2027-redesign`

FairEval is a preference-conditioned benchmark for evaluating fairness in LLM-based recommendation with personality awareness. The central design principle is simple: **recommendation change is sensitivity; harmfulness requires consequence evidence**.

The ECIR 2027 redesign separates demographic counterfactual effects from beneficial personalization, grounds personality in measured Big Five profiles, freezes prompt/cue/order controls before execution, and reports hosted/API and local/open-weight evidence in separate strata.

## Research questions

- **RQ1 — Demographic consequence.** Holding observed preferences, candidates, task, and prompt structure fixed, when demographic context changes an LLM ranking, does the change improve held-out utility, leave utility effectively unchanged, or create a harmful counterfactual utility/exposure disparity?
- **RQ2 — Grounded personality value.** Does measured Big Five personality provide user-specific recommendation value beyond observed preference history, and does that value survive shuffled-profile and one-trait counterfactual controls?
- **RQ3 — Reliability and generalization.** How stable are FairEval conclusions across datasets, model families, task paraphrases, cue realizations, candidate order, ranking cutoffs, and repeated generations?
- **RQ4 — Personalization-preserving mitigation.** Can instruction-based and counterfactual re-ranking interventions reduce harmful identity-conditioned gaps while preserving overall utility and beneficial personality-driven personalization?

## Benchmark design

The real-world benchmark contains six datasets spanning personality-aware recommendation, demographic fairness, and domain generalization. FairSynth-360 is a project-generated 360-user controlled sanity benchmark whose synthetic A/B/C identity is independent of relevance by construction. Real and synthetic results are always reported separately.

The primary matched conditions are `C0` preference history only, `C1` observed demographic context, `C2` matched one-field demographic counterfactual, `C3` true measured personality, `C4` deterministic whole-profile derangement, and `C5` one measured Big Five trait changed while the other four stay fixed.

Candidates, history, task structure, and candidate ordering are frozen across matched conditions. The model must return exactly `K` unique candidate IDs from the supplied candidate set in structured JSON.

## Metrics and inference

Primary utility uses nDCG@10 and Recall@10. RBO and Jaccard measure ranking sensitivity only. Fairness/personality estimands include Counterfactual Utility Gap (CUG), Group Utility Disparity (GUD), Counterfactual Exposure Gap (CEG) only where complete auditable metadata exists, Invalid Output Disparity (IOD), and Personality Value Added (PVA).

User is the primary unit of inference. Repeated generations are averaged within user-condition before pairing. The main inference stack uses paired bootstrap confidence intervals, paired sign-flip permutation tests, Wilcoxon sensitivity analysis, matched rank-biserial effect size, and Holm correction inside pre-registered families.

Persistent invalid output after one permitted format-only repair receives zero primary system utility and remains an observable outcome; it is never regenerated repeatedly until valid.

## Repository structure

```text
configs/                 frozen study/model/dataset/prompt contracts
src/faireval/            benchmark, execution, provider and analysis code
scripts/                 acquisition, freeze, planning, execution and paper tools
data/                     project-owned + locally acquired data layout
results/                  run plans, raw outputs, analysis artifacts and figures
paper/                    LNCS/ECIR manuscript source
provenance/               environment and execution records
```

## Environment

Create a Python 3.10+ environment and install the project:

```bash
python -m pip install -e ".[analysis,dev,providers]"
```

Never commit API keys. Copy `.env.example` to `.env` and fill only the credentials required for the execution path you are using.

## Dataset safety

Third-party raw releases are not silently downloaded or substituted. Before a paid real-world pilot, acquire each exact upstream release under its documented terms, place it under the configured raw path, compute and record its hash, freeze deterministic benchmark instances, and verify hashes before compiling a run plan. FairEval never infers protected attributes from names, free text, embeddings, ZIP codes, or model guesses.

## Deterministic data preparation

```bash
faireval prepare \
  --dataset movielens_1m \
  --raw-dir data/raw/movielens_1m \
  --output-dir data/frozen/movielens_1m \
  --users 20 \
  --candidate-set-size 50 \
  --max-history-items 20 \
  --seed 1729

faireval verify-freeze --output-dir data/frozen/movielens_1m
```

The freeze manifest records instance count, canonical instance SHA-256, adapter/version metadata, and split/candidate-generation provenance.

## Immutable run plans

Each JSONL cell contains a deterministic `cell_id`; the plan file itself is hashed and linked to a manifest. Execution verifies the plan hash and every cell ID before constructing a provider. Resume semantics are exact: already persisted `planned_cell_id` values are not regenerated, and invalidity remains an experimental outcome.

The six-real-dataset hosted core plan remains gated on exact third-party dataset freezes. The API-only evidence layer that can run immediately is the project-owned FairSynth control.

## Hosted API execution — 200/250 RMB freeze

The current black-box phase uses the Zhizengzeng OpenAI-compatible gateway with a single gateway key. The gateway/model freeze is in `configs/models.yaml`; spend policy is in `configs/hosted_budget.yaml`.

Current hosted panel:

- OpenAI: `gpt-5.6-terra`
- Anthropic: `claude-sonnet-5`
- Google: `gemini-3.8-flash`
- DeepSeek: `deepseek-v4.1-flash`
- Alibaba Qwen: `qwen3.8-2.4t-a95b`
- Meta: `llama-4-maverick`

Hosted environment:

```dotenv
FAIREVAL_HOSTED_GATEWAY=zhizengzeng
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
ZZZ_API_KEY=<secret>
```

The runner itself performs a live `GET /v1/models` exact-ID gate before paid generation. If an exact frozen ID is absent, execution is blocked before spend; a nearby model is never substituted silently.

The hosted budget is frozen as a **200 RMB normal stop target**, **250 RMB non-bypassable emergency ceiling**, and **2 RMB pre-cell reserve**. A 50 RMB buffer therefore remains between ordinary stopping and the absolute ceiling. Spend is reconciled from Zhizengzeng account-balance movement rather than hand-estimated token counts.

### Immediate hosted FairSynth plan

Create/verify the 360-user project-owned freeze:

```bash
python scripts/build_fairsynth360.py \
  --output-dir data/frozen/fairsynth360 \
  --users 360 \
  --candidate-set-size 30 \
  --max-history-items 8 \
  --seed 1729
```

Compile the default deterministic 120-user A/B/C-balanced hosted plan:

```bash
python scripts/plan_hosted_fairsynth.py \
  --freeze-root data/frozen \
  --output-dir results/plans/hosted-fairsynth-budget-v1 \
  --users 120 \
  --repetitions 3 \
  --seed 1729
```

This yields 12,960 immutable planned cells: 120 users × 6 FairSynth conditions × 6 hosted families × 3 repetitions. The plan may be truncated by the budget, but its order is fixed independently of results.

Dry-run first:

```bash
python scripts/run_hosted_budgeted.py \
  --plan-dir results/plans/hosted-fairsynth-budget-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/runs/hosted-fairsynth-budget-v1.jsonl
```

Run a 12-cell paid canary:

```bash
python scripts/run_hosted_budgeted.py \
  --plan-dir results/plans/hosted-fairsynth-budget-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/runs/hosted-fairsynth-budget-v1.jsonl \
  --ledger results/budget/hosted_zzz_v1.json \
  --target-rmb 200 \
  --hard-cap-rmb 250 \
  --request-reserve-rmb 2 \
  --max-cells 12 \
  --execute
```

After auditing the canary, rerun the same command without `--max-cells` to resume. The runner cannot accept a target above 200 RMB or a hard ceiling above 250 RMB.

## Local/open-weight transparency track

The repository retains the Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct transparency track for later reproducibility/white-box analysis. It is optional during the current hosted-only API phase and is not pooled with hosted-model claims.

## Analysis and direct paper synchronization

Hosted FairSynth numerical results use this artifact-only path:

```text
hosted JSONL
→ run-log audit
→ FairSynth paired inference
→ paper/generated/fairsynth_hosted_table.tex
→ paper/results_contract_table.tex
→ Overleaf ZIP
```

One command performs the post-run pipeline:

```bash
python scripts/finalize_hosted_fairsynth.py
```

The generated six-model table reports synthetic identity and synthetic personality paired nDCG@10 differences, bootstrap confidence intervals, Holm-adjusted permutation p-values, and paired-user counts. It is automatically included in the paper and the exported Overleaf ZIP only when the audited artifact exists.

For the full real-world study, the existing result-table/figure renderers remain the canonical path. Until audited artifacts exist, corresponding paper cells remain explicitly pending; empirical numbers are never typed into the manuscript manually.

## Reproducibility checks

```bash
python scripts/check_config_consistency.py
python scripts/check_paper_source.py
python scripts/check_result_table_contracts.py
python scripts/check_pilot_readiness.py
pytest -q
```

`check_pilot_readiness.py --strict` intentionally remains non-zero for the six-real-dataset confirmatory panel until exact third-party release locks and local raw paths are complete. That does not block the separately scoped project-owned hosted FairSynth sanity run.

## Reporting discipline

FairEval does not claim that one model is fairer than another, that measured personality necessarily improves recommendation, or that a mitigation works until the frozen pipeline produces validated evidence. Real-world, synthetic, hosted, and optional local evidence retain their declared scope throughout analysis and the paper.
