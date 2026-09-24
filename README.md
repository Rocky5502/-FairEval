# FairEval

> **ECIR 2027 final paper branch:** `ecir-2027-final-results`. The frozen V7 local experiment ran at commit `83d7ac443a...` and completed 2,880/2,880 audited cells.

FairEval is a preference-conditioned benchmark for evaluating fairness in LLM-based recommendation with personality awareness. The central design principle is simple: **recommendation change is sensitivity; harmfulness requires consequence evidence**.

The ECIR 2027 redesign separates demographic counterfactual effects from beneficial personalization, grounds personality in measured Big Five profiles, freezes prompt/cue/order controls before execution, and reports hosted/API and local/open-weight evidence in separate strata.

## Final V7 local experiment status

The frozen V4 campaign and failed V5/V6 protocol canaries remain preserved as
provenance. V7 introduced deterministic candidate-selection handles, passed the
unchanged sealed 95% per-model canary gate, and was then used for the final
deadline-bounded local FairSynth matrix.

Final V7 main run:

- 120 canary-excluded FairSynth users;
- 6 registered conditions;
- 2 seeded repetitions;
- 2 frozen local model families;
- 2,880 planned and observed cells;
- final integrity audit: PASS;
- Phi-3.5-mini semantic validity: 1,440/1,440;
- Qwen2.5-7B semantic validity: 1,371/1,440 (95.21%).

Persistent Qwen failures are retained as system behavior; there is no generative
repair or outcome-dependent rerun. The final evidence/provenance summary is
`provenance/V7_FINAL_RESULTS_2026-09-24.md`.


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

Persistent invalid output that remains semantically invalid after deterministic envelope parsing receives zero primary system utility and remains an observable outcome; canonical V5 uses one generation per cell and no generative/LLM repair.

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

`configs/dataset_releases.yaml` intentionally remains pre-freeze until the exact files exist locally. **Do not invent missing checksums or mark a license reviewed without checking the downloaded release.**

## Pre-execution scientific seal

Before any paid hosted generation or local-model inference, build the zero-call seal:

```bash
python scripts/build_preexecution_seal.py \
  --output-dir results/preexecution/seal-v1
```

This step loads no model weights and makes no hosted generation calls. It regenerates the canonical FairSynth freeze, compiles the deterministic hosted and white-box FairSynth plans, renders pre-result paper contracts/figures, runs configuration/manuscript checks, hashes the tracked scientific specification, records unresolved real-dataset blockers, and creates a pre-result anonymous Overleaf ZIP.

Both real launchers require the seal's Git commit and plan SHA to match the current checkout. A scientific-source or plan change therefore requires a new seal/version rather than silently continuing an old run log.

## Immutable run plans

Each JSONL cell contains a deterministic `cell_id`; the plan file itself is hashed and linked to a manifest. Execution verifies the plan hash and every cell ID before constructing a provider. Resume semantics are exact: already persisted `planned_cell_id` values are not regenerated, and invalidity remains an experimental outcome.

The six-real-dataset hosted core plan remains gated on exact third-party dataset freezes. The immediately executable project-owned control is FairSynth-360.

## Hosted API execution — lean 1,080-call FairSynth campaign

The black-box track uses the Zhizengzeng OpenAI-compatible gateway with a single gateway key. The gateway/model freeze is in `configs/models.yaml`; spend policy is in `configs/hosted_budget.yaml`.

Current hosted panel:

- OpenAI: `gpt-5.6-terra`
- Anthropic: `claude-sonnet-5`
- Google: `gemini-3.8-flash`
- DeepSeek: `deepseek-v4.1-flash`
- Alibaba Qwen: `qwen3.8-max`
- Meta: `llama-4-maverick`

Hosted environment:

```dotenv
FAIREVAL_HOSTED_GATEWAY=zhizengzeng
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
ZZZ_API_KEY=<secret>
```

The runner performs a live `GET /v1/models` exact-ID gate before paid generation. The 2026-09-18 experiment-key preflight rejected the earlier Qwen pre-run ID before any paid call; the manifest was then versioned to the actually exposed `qwen3.8-max`. Future mismatches are handled the same way: block first, version explicitly, never substitute silently.

The project-wide safety ceiling remains a **200 RMB normal stop target**, **250 RMB client-side emergency stop threshold**, and **2 RMB pre-cell reserve**; these are ceilings, not a spending objective. After six one-cell operational canaries (0.3041 RMB total observed balance movement), and before analyzing recommendation outcomes, the hosted FairSynth campaign was versioned to a stricter **65 RMB runtime target / 75 RMB emergency threshold**.

The canonical hosted FairSynth plan now has **1,080 immutable calls**: 30 deterministically selected balanced users (10 per A/B/C identity group) × 6 registered FairSynth conditions × 6 hosted families × 1 generation. FairSynth is an auxiliary controlled sanity layer, so this design preserves all registered synthetic contrasts while avoiding redundant hosted repeats. Budget truncation is reported as incomplete planned coverage; the execution order never adapts to observed results.

Dry-run first:

```bash
python scripts/run_hosted_budgeted.py \
  --plan-dir results/plans/hosted-fairsynth-lean-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/runs/hosted-fairsynth-lean-v1.jsonl \
  --preexecution-seal results/preexecution/seal-v1/PREEXECUTION_SEAL.json
```

Real execution additionally requires the exact checked-out SHA:

```bash
SHA=$(git rev-parse HEAD)
python scripts/run_hosted_budgeted.py \
  --plan-dir results/plans/hosted-fairsynth-lean-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/runs/hosted-fairsynth-lean-v1.jsonl \
  --ledger results/budget/hosted_zzz_lean_v1.json \
  --preexecution-seal results/preexecution/seal-v1/PREEXECUTION_SEAL.json \
  --target-rmb 65 \
  --hard-cap-rmb 75 \
  --request-reserve-rmb 2 \
  --max-cells 12 \
  --code-commit-sha "$SHA" \
  --execute
```

## Completed local/open-weight white-box stratum

The repository freezes Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct as a **required, separately reported replication stratum**. Hosted and local effects are never pooled merely because they answer the same RQ, and local token-score diagnostics are explicitly auxiliary and uncalibrated.

The submission-time FairSynth V7 plan contains exactly **2,880 generations**:

```text
120 canary-excluded users × 6 conditions × 2 seeded repetitions × 2 frozen models
```

This complete matrix has finished and passed its integrity audit. The earlier
12,960-cell geometry remains historical planning provenance only.

The canonical local execution profile is a single 16 GB RTX 5070 Ti with one model loaded at a time using fixed bitsandbytes NF4 4-bit quantization and bfloat16 compute. On the pinned Transformers 4.44.2 stack, both Qwen2.5 and Phi-3.5 keep KV caching enabled; Phi uses eager attention and may emit a `seen_tokens` deprecation warning, which is not treated as a runtime failure. These execution choices are frozen in `configs/local_models.yaml`, persisted into the immutable plan, and rechecked against the runtime provider.

The two model families can be run individually through `scripts/run_whitebox_family.py`, or end-to-end through `scripts/run_whitebox_full.py`. Real GPU execution requires the exact checked-out Git SHA and matching pre-execution seal. The older `scripts/run_whitebox_campaign.py` generic-prompt engineering runner is retired: its Qwen/Mistral/Phi outputs are smoke artifacts only and must not be used as ECIR empirical results because they did not execute the immutable six-condition FairEval plan.

`scripts/finalize_whitebox.py` refuses partial/mixed coverage, merges both audited canonical family logs in immutable plan order, computes the declared diagnostics, renders `paper/generated/whitebox_summary_table.tex`, and rebuilds the anonymous Overleaf bundle.

The completed V7 execution path is `scripts/run_whitebox_v7_lean.py`, with
`scripts/audit_whitebox_v7_lean.py` as the final integrity gate. No additional
GPU inference is required for the current ECIR submission.

Once all real-world release locks are complete, a separately versioned campaign extends the same two local models to the executable RQ1/RQ2 core, registered RQ3 robustness factors, and RQ4 mitigation path.

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

One command performs the hosted post-run pipeline:

```bash
python scripts/finalize_hosted_fairsynth.py
```

The full local path is similarly wrapped by `scripts/finalize_whitebox.py`. Until audited artifacts exist, corresponding paper cells remain explicitly pending; empirical numbers are never typed into the manuscript manually.

## No-new-call secondary analysis

After the audited V7 run, FairEval can derive one additional descriptive figure
without loading a model or making an API request. It summarizes condition-level
utility and the per-user paired-effect distributions from the already frozen
`user_condition.jsonl`, `identity_pairs.jsonl`, and `personality_pairs.jsonl`.

```bash
python scripts/analyze_v7_secondary.py \
  --user-condition results/analysis/fairsynth-v7-lean/user_condition.jsonl \
  --identity-pairs results/analysis/fairsynth-v7-lean/identity_pairs.jsonl \
  --personality-pairs results/analysis/fairsynth-v7-lean/personality_pairs.jsonl \
  --output-dir results/analysis/fairsynth-v7-secondary \
  --paper-figure paper/figures/v7_secondary_profiles.pdf
```

The analysis is explicitly descriptive: it creates no new confirmatory hypothesis
family and never changes the frozen primary inference.

## ECIR submission closeout

The current ECIR submission has a separate final scope manifest:
`configs/submission_scope_ecir2027.yaml`. It freezes the completed evidence as
the 2,880-cell local V7 FairSynth experiment plus the 99-cell hosted operational
pilot. The six real-world datasets remain explicitly deferred and unclaimed;
their missing third-party release locks are **not** treated as a submission
failure and are never fabricated.

Run one final closeout audit:

```bash
python scripts/check_submission_closeout.py
```

A PASS means no additional GPU inference is required for the current paper,
all paper-facing artifacts and guards are present, the test/preflight suite
passes, and a fresh anonymous Overleaf ZIP has been built. The stricter
`check_pilot_readiness.py --strict` may correctly remain non-zero because it
answers a different question: whether the deferred six-dataset paid campaign is
ready to launch.

## Reproducibility checks

```bash
python scripts/check_config_consistency.py
python scripts/check_paper_source.py
python scripts/check_result_table_contracts.py
python scripts/check_pilot_readiness.py
python scripts/build_preexecution_seal.py
pytest -q
```

`check_pilot_readiness.py --strict` intentionally remains non-zero for the six-real-dataset confirmatory panel until exact third-party release locks and local raw paths are complete. That does not block the separately scoped project-owned FairSynth hosted/local control experiments.

## Reporting discipline

FairEval does not claim that one model is fairer than another, that measured personality necessarily improves recommendation, or that a mitigation works until the frozen pipeline produces validated evidence. Real-world, synthetic, hosted, and local evidence retain their declared scope throughout analysis and the paper.
