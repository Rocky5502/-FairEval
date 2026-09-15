# FairEval — ECIR 2027 Reboot

**Target submission title:** *FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness*

This repository is a clean-room scientific redesign of FairEval for the **ECIR 2027 Full Paper Track**. It is not a cosmetic extension of earlier work: the evaluation target, datasets, prompt controls, statistical design, reliability analysis, local transparency track, controlled synthetic sanity benchmark, and mitigation study are built around the distinction between useful personalization and harmful identity-conditioned disparity.

## Core scientific change

FairEval separates four quantities that should not be conflated:

1. **Behavioral shift** — did the ranking change?
2. **Held-out utility consequence** — did the change improve, preserve, or harm recommendation quality?
3. **Counterfactual demographic effect** — does demographic context change utility/exposure when preference evidence and candidates are held fixed?
4. **Grounded personality value** — does a measured Big Five profile add user-specific recommendation value beyond observed preferences and a shuffled-profile negative control?

A changed list is therefore **not automatically labeled unfair**. RBO/Jaccard are sensitivity diagnostics; fairness claims are tied to consequence-aware paired utility/failure/exposure evidence.

## Four research questions

- **RQ1 — Consequence of demographic context.** Holding observed preferences, candidates, task, prompt structure, and generation settings fixed, when demographic context changes a ranking, does the change improve utility, leave utility effectively unchanged, or create a harmful counterfactual utility/exposure disparity?
- **RQ2 — Grounded personality value.** Does measured Big Five personality provide user-specific recommendation value beyond observed preference history, and does that value survive shuffled-profile and one-trait controls?
- **RQ3 — Reliability and generalization.** How stable are conclusions across six real-world datasets, the separately reported FairSynth-360 control benchmark, six hosted/API families, two local open-weight models, prompt/cue formulations, candidate order, ranking cutoffs, and repeated generations?
- **RQ4 — Personalization-preserving mitigation.** Can identity-irrelevance prompting and contextual PAIR re-ranking reduce harmful identity-conditioned gaps while preserving recommendation utility and beneficial personalization?

## Benchmark structure

### Six real-world datasets

**Personality-grounded track**

- **GroupLens Personality 2018** — movies + measured Big Five
- **Music Master / BFI-2** — music + BFI/BFI-2 domains/facets
- **REASONER** — short video + CBF-PI-15 Big Five responses

**Demographic/generalization track**

- **MovieLens-1M** — ratings + dataset-supplied demographic fields
- **Last.fm-1K** — listening histories + dataset-native profile fields where available/licensed
- **MIND** — news clicks + logged impression candidates; preference-only in the core design

Protected attributes are never inferred from names, text, ZIP codes, embeddings, or model guesses.

### FairSynth-360 — project-generated controlled sanity benchmark

FairSynth-360 is generated entirely from versioned code in this repository:

- 360 synthetic users/tasks;
- 120-item synthetic catalog;
- balanced synthetic identity labels `A/B/C` — 120 users each;
- identity generated independently of preferences and relevance;
- synthetic OCEAN-like vectors partially informative about a latent preference vector;
- relevance determined only by latent preference × item features;
- no external or personal data.

Its identity labels are **not demographic categories** and its OCEAN vectors are **not measured human personality**. FairSynth inference is never pooled with RQ1 observed-demographic or RQ2 measured-psychometric inference.

Build it with:

```bash
python scripts/build_fairsynth360.py
```

## Eight model configurations: six hosted + two local

### Hosted/API core

- OpenAI — GPT-5.6 Terra
- Anthropic — Claude Sonnet 5
- Google — Gemini 3.8 Flash
- DeepSeek — V4.1 Flash (`deepseek-flash`)
- Alibaba — Qwen3.8-Max-0902
- Meta — Llama 4 Maverick; exact hosting provider/revision still must be frozen

`configs/models.yaml` freezes reasoning/thinking policy, sampling policy, and provider-specific output-token semantics.

### Local open-weight transparency track

- **Qwen2.5-7B-Instruct** — `Qwen/Qwen2.5-7B-Instruct`, Apache-2.0
- **Phi-3.5-mini-instruct** — `microsoft/Phi-3.5-mini-instruct`, MIT

The canonical local path uses direct Transformers inference rather than an API wrapper, which allows FairEval to log auxiliary token-level generation-score diagnostics:

- mean generated-token log-probability;
- generated-token NLL;
- generated-token perplexity;
- mean top-1/top-2 generation-score margin.

These signals are **exploratory and uncalibrated**. They are not treated as a probability that a recommendation is correct and are not directly compared with unavailable internal scores from hosted APIs.

Preferred single-GPU target: **NVIDIA GeForce RTX 5090, 32 GB VRAM**, one model loaded at a time. The repo does not assume an unofficial “RTX 5090 Ti.” See `docs/LOCAL_GPU_TRACK.md`.

Install the local track after installing a CUDA-matched PyTorch build:

```bash
pip install -r requirements-local-gpu.txt
pip install -e .
python scripts/check_local_gpu.py
```

Before the real local pilot, freeze exact Hugging Face revisions in `QWEN25_LOCAL_REVISION` and `PHI35_LOCAL_REVISION`.

## Prompt design

Prompt design is a first-class experimental variable:

- neutral audit system prompt for RQ1--RQ3;
- separately named identity-irrelevance mitigation prompt for RQ4;
- three pre-registered task paraphrases;
- explicit demographic cue representations;
- fixed top-level fields with `unspecified` null conditions;
- psychometric instrument metadata accompanying normalized OCEAN values;
- deterministic per-user candidate-order seed reused inside matched counterfactuals;
- exact-K candidate-ID JSON output with no chain-of-thought/rationale request;
- prompt SHA-256 logged for every run.

## Factorized experiment design

The study avoids an uninterpretable Cartesian product:

- **RQ1:** MovieLens-1M + Last.fm-1K, observed vs matched demographic counterfactuals;
- **RQ2:** three measured-personality datasets, true vs shuffled plus one-trait controls;
- **RQ3:** one-factor-at-a-time prompt/cue/order/K/stochasticity robustness;
- **RQ4:** unmitigated audit vs identity-irrelevance prompting vs contextual PAIR;
- **local replication:** Qwen2.5/Phi-3.5 reuse frozen real-world core tasks;
- **FairSynth-360:** separate controlled identity/personality sanity analysis.

The real-data sample-size plan uses a variance/invalidity pilot, then freezes confirmatory N without using the pilot treatment-effect mean to chase significance.

## Metrics and inference

**Recommendation utility:** nDCG@10, Recall@10; MRR secondary.

**Counterfactual consequence/fairness:** signed/absolute CUG, GUD when justified, CEG only when complete top-K group metadata exists, and IOD.

**Personality:** PVA = utility(true measured profile) − utility(shuffled profile), plus one-trait controls.

**Sensitivity only:** RBO/Jaccard and legacy PAFS-style quantities.

Persistent invalid responses remain outcomes. After the one permitted format-only repair fails, primary end-to-end utility is zero; valid-only utility is a labeled sensitivity analysis and IOD is reported separately.

`configs/analysis.yaml` freezes:

- user as the primary inference unit;
- averaging repetitions within user-condition before pairing;
- paired permutation as the primary matched test;
- paired bootstrap confidence intervals;
- Wilcoxon signed-rank sensitivity;
- matched rank-biserial effect sizes;
- Holm correction within pre-registered `RQ × metric × contrast` families.

FairSynth uses separate inference families. Its two alternative A/B/C counterfactual labels are reduced to **one mean counterfactual outcome per user** before inference, avoiding pseudo-replication.

## Contextual PAIR mitigation

For candidate item `i`, preference-only support `b(i)`, focal identity-context support `r_a(i)`, and identity contexts `A`:

```text
Fit_a(i)         = alpha * b(i) + (1-alpha) * r_a(i)
CFInstability(i) = Var_{a' in A}[r_a'(i)]
PAIR_a(i)        = Fit_a(i) - lambda * CFInstability(i)
```

`alpha` and `lambda` are selected on validation users only. This focal-context formulation avoids the degenerate earlier idea of averaging all identity-conditioned rankings into one common ranking, which would mechanically force counterfactual output disparity toward zero.

## Core execution path

```bash
python scripts/audit_run_log.py \
  --output-jsonl results/raw/core-v1.jsonl \
  --plan-dir results/plans/core-v1

python scripts/analyze_core.py \
  --output-jsonl results/raw/core-v1.jsonl \
  --plan-dir results/plans/core-v1 \
  --freeze-root data/frozen \
  --output-dir results/analysis/core-v1
```

## Local/FairSynth execution path

Build the synthetic freeze:

```bash
python scripts/build_fairsynth360.py
```

Compile the two-model local plan over the real freezes plus FairSynth:

```bash
python scripts/plan_local_open_weight.py \
  --freeze-root data/frozen \
  --output-dir results/plans/local-open-weight-v1 \
  --fairsynth-users 360 \
  --repetitions 3
```

Dry-run a local family without loading/calling the model:

```bash
python -m faireval.cli execute-plan \
  --plan-dir results/plans/local-open-weight-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/raw/local-open-weight-v1.jsonl \
  --family qwen25_local \
  --max-cells 10
```

Add `--execute --code-commit-sha <commit>` only after reviewing the frozen plan and strict GPU/revision preflight.

Analyze synthetic sanity results separately:

```bash
python scripts/analyze_fairsynth360.py \
  --output-jsonl results/raw/local-open-weight-v1.jsonl \
  --plan-dir results/plans/local-open-weight-v1 \
  --freeze-root data/frozen
```

Analyze local internal scores separately:

```bash
python scripts/analyze_local_whitebox.py \
  --output-jsonl results/raw/local-open-weight-v1.jsonl \
  --plan-dir results/plans/local-open-weight-v1 \
  --freeze-root data/frozen
```

## Professional paper assets

The repository contains reproducible vector-PDF methodology figures generated by `scripts/build_paper_figures.py`, plus machine-generated result figures/tables after experiments. The related-work coverage table is backed by an evidence ledger and uses conservative Yes / Partial / Not-established coding; it is not an empirical superiority claim.

The manuscript includes:

- related-work design-coverage table;
- combined real/synthetic dataset + hosted/local model table;
- RQ→estimand→inference matrix;
- multi-panel RQ1–RQ4 + FairSynth/local-white-box results contract.

## Repository layout

```text
configs/          core + local/synthetic model/dataset/study/analysis manifests
src/faireval/     adapters, providers, prompts, runner, audit, analysis, mitigation
data/publishable/ project-generated dataset documentation (raw third-party data excluded)
tests/            invariants, providers, plans, analysis, statistics, synthetic/local guards
scripts/          setup, preflight, planning, execution support, analysis, paper generation
docs/             runbooks, literature/novelty/submission guards, local GPU track
paper/            anonymous LNCS manuscript, audited tables, vector-PDF figures
results/          generated outputs; raw/large results are not committed
```

## Integrity rules

- No empirical number is invented or hand-entered into the final paper.
- Pilot and confirmatory outputs remain labeled/separated.
- Invalid generations remain observable outcomes.
- No prompt/model/dataset/K/hyperparameter is selected after viewing test outcomes and then presented as confirmatory.
- FairSynth is never pooled with real demographic/psychometric inference.
- Local token-score diagnostics are not labeled calibrated uncertainty.
- Hosted vs local results are stratified rather than interpreted as a causal open-vs-closed comparison.
- Exact local Hugging Face revisions are frozen before local execution.
- Tables/figures are generated from frozen result artifacts with input/config/code hashes.

## ECIR constraints

- Abstract registration: **21 September 2026, 23:59 GMT**.
- Full-paper deadline: **5 October 2026, 23:59 GMT**.
- Springer LNCS; maximum **12 pages plus references**; appendices count inside the 12-page limit.
- Double-blind review.

See `docs/ECIR2027_SUBMISSION_COMPLIANCE.md` before export.

## Current status

- consequence-aware scientific redesign: implemented
- six real-world adapters/configs: implemented
- FairSynth-360 generator/adapter/freeze/tests: implemented
- six hosted provider layer: implemented; final Llama host still pending freeze
- Qwen2.5-7B + Phi-3.5-mini direct local Transformers providers: implemented
- local hardware/revision preflight: implemented
- controlled prompt/cue/order machinery: implemented + tested
- immutable core/local plan and execution provenance: implemented
- FairSynth user-level sanity estimands + separate inference: implemented
- local token-score exploratory analysis: implemented
- RQ1/RQ2/RQ3 analysis infrastructure: implemented
- contextual PAIR formulation/code: implemented; RQ4 end-to-end result builder remains to be finalized
- professional methodology figures + multi-layer paper tables: implemented
- empirical result values: intentionally **not populated** until execution
- next execution milestone: freeze six real dataset releases, freeze exact Llama host, freeze exact Qwen2.5/Phi-3.5 HF revisions, run strict readiness checks, then execute the variance/invalidity pilot
