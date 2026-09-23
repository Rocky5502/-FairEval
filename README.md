
## V5 output-protocol recovery

The frozen V4 local white-box campaign remains immutable. The separately
versioned V5 recovery removes generative format repair, uses deterministic
envelope-only parsing, and must pass a sealed 144-cell FairSynth canary before
any full rerun.

Recovery artifacts and entry points:

- forensic V4 reparse: `scripts/analyze_whitebox_v4_forensic.py` (post-hoc only);
- V5 canary plan: `scripts/build_whitebox_v5_canary.py`;
- V5 pre-execution seal: `scripts/build_whitebox_v5_seal.py`;
- V5 canary execution: `scripts/run_whitebox_v5_canary.py`;
- V5 promotion audit: `scripts/audit_whitebox_v5_canary.py`;
- frozen V4 boundary: `provenance/V4_OUTPUT_PROTOCOL_FAILURE_2026-09-23.md`.

The scale-up rule is outcome-independent and predeclared: semantic exact-K
candidate validity must be at least 95% **for each local model** over all 72
canary cells, with zero candidate-ID mutation, zero parser ambiguity, and all
provenance gates passing. A failed subgate means diagnose first; do not scale.

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

## Full-scale local/open-weight white-box stratum

The repository freezes Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct as a **required, separately reported replication stratum**. Hosted and local effects are never pooled merely because they answer the same RQ, and local token-score diagnostics are explicitly auxiliary and uncalibrated.

The canonical FairSynth white-box plan contains exactly **12,960 generations**:

```text
360 users × 6 conditions × 3 seeded repetitions × 2 frozen models
```

The canonical local execution profile is a single 16 GB RTX 5070 Ti with one model loaded at a time using fixed bitsandbytes NF4 4-bit quantization and bfloat16 compute. On the pinned Transformers 4.44.2 stack, both Qwen2.5 and Phi-3.5 keep KV caching enabled; Phi uses eager attention and may emit a `seen_tokens` deprecation warning, which is not treated as a runtime failure. These execution choices are frozen in `configs/local_models.yaml`, persisted into the immutable plan, and rechecked against the runtime provider.

The two model families can be run individually through `scripts/run_whitebox_family.py`, or end-to-end through `scripts/run_whitebox_full.py`. Real GPU execution requires the exact checked-out Git SHA and matching pre-execution seal. The older `scripts/run_whitebox_campaign.py` generic-prompt engineering runner is retired: its Qwen/Mistral/Phi outputs are smoke artifacts only and must not be used as ECIR empirical results because they did not execute the immutable six-condition FairEval plan.

`scripts/finalize_whitebox.py` refuses partial/mixed coverage, merges both audited canonical family logs in immutable plan order, computes the declared diagnostics, renders `paper/generated/whitebox_summary_table.tex`, and rebuilds the anonymous Overleaf bundle.

Preferred guarded one-command execution:

```bash
python scripts/run_whitebox_smart.py --execute
```

This launcher first checks the exact local package/GPU environment, builds a fresh scientific seal, runs one real canonical cell per family as a non-wasted canary, resumes directly into the full 12,960-cell campaign only if both canaries pass, and then audits/analyzes the completed logs and rebuilds the Overleaf bundle automatically.

Lower-level execution remains available through `scripts/run_whitebox_full.py` when manual control is needed.

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
