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
- **Last.fm-1K** — music histories + dataset-native profile fields where available/licensed
- **MIND** — news click histories + logged impression candidates; the frozen core plan is preference-only and makes no observed-demographic claim from MIND

Every adapter records which fields are observed, derived, or synthetically counterfactualized. Protected attributes are never inferred from names, text, ZIP codes, or embeddings.

## Six LLM families

The pre-run model manifest uses one frozen representative from each family:

- OpenAI — GPT-5.6 Terra
- Anthropic — Claude Sonnet 5
- Google — Gemini 3.8 Flash
- DeepSeek — V4.1 Flash (`deepseek-flash`)
- Alibaba — Qwen3.8-Max-0902
- Meta — Llama 4 Maverick checkpoint; exact serving provider still must be frozen before the pilot

`configs/models.yaml` records each family’s reasoning/thinking policy, sampling policy, and provider-specific output-token request field. Current OpenAI Chat Completions uses `max_completion_tokens`; Anthropic/DeepSeek/Qwen and the provisional OpenAI-compatible Llama host use `max_tokens`; Gemini Interactions uses `max_output_tokens`. Provider adapters log what was actually applied.

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

- **RQ1:** MovieLens-1M and Last.fm-1K with paired observed/counterfactual demographic interventions;
- **RQ2:** the three measured-personality datasets with true, shuffled, and one-trait controls;
- **RQ3:** pre-registered prompt/cue/order/stochasticity robustness subsets across all six families/datasets;
- **RQ4:** demographic datasets with instruction baselines and PAIR re-ranking;
- **MIND:** candidate-constrained domain/reliability evaluation in the frozen core plan; any future synthetic identity stress test requires a separately versioned plan.

The sample-size plan uses a variance/invalidity pilot, then freezes confirmatory N without using the pilot treatment-effect mean to chase significance.

## Primary metrics

**Recommendation utility:** nDCG@10, Recall@10; MRR as secondary.

**Counterfactual fairness/consequence:** signed and absolute Counterfactual Utility Gap (CUG), Group Utility Disparity when justified, Counterfactual Exposure Gap only when complete auditable item-group metadata exists, and Invalid Output Disparity.

**Personality:** Personality Value Added (PVA), comparing the true measured profile with a matched shuffled-profile control, plus pre-registered one-trait interventions.

**Sensitivity only:** RBO/Jaccard and the old PAFS-style quantity are diagnostics, not primary fairness definitions.

Persistent invalid responses remain outcomes. The frozen primary end-to-end utility policy assigns zero nDCG/Recall/MRR after the one allowed format-only repair fails; valid-only utility is reported only as a sensitivity analysis, while invalid-output disparity is reported separately. This prevents complete-case filtering from making unreliable models look artificially strong.

## Statistical analysis

`configs/analysis.yaml` freezes the inferential policy before results:

- repeated generations are averaged within user-condition before inference;
- paired permutation is the primary test;
- user-level paired bootstrap provides confidence intervals;
- Wilcoxon signed-rank is a sensitivity analysis;
- matched rank-biserial is reported as an effect size;
- Holm correction is applied within pre-registered `RQ × metric × contrast` families;
- effect magnitude and uncertainty are reported alongside p-values.

The raw-to-analysis path is executable rather than manual:

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

The analysis builder audits response hashes and plan linkage, scores frozen held-out relevance, aggregates repetitions at the user level, builds RQ1/RQ2 paired estimands, executes the frozen inference stack, and writes hashes for every generated analysis artifact. Paper numbers are not hand-entered.

## Mitigation

RQ4 compares the unmitigated audit condition with:

1. identity-irrelevance instruction;
2. a counterfactually fair prompting baseline;
3. **PAIR — Preference-Aligned Identity Re-ranking**, whose trade-off parameter is chosen on validation data only.

Success requires reducing harmful utility/exposure gaps **without merely forcing rankings to look similar**. Utility--fairness Pareto frontiers are reported instead of selecting a post-hoc operating point.

## Professional paper assets

The repository contains reproducible vector-PDF methodology figures generated by `scripts/build_paper_figures.py`:

- `paper/figures/faireval_framework.pdf`
- `paper/figures/faireval_conditions.pdf`
- `paper/figures/faireval_evaluation_pipeline.pdf`

CI regenerates and validates these PDF assets. The 16-row related-work design-coverage table is backed by `docs/LITERATURE_COMPARISON_AUDIT.md`; red checks indicate explicit documented design coverage, gray markers indicate partial/adjacent coverage, and black crosses mean the criterion was not established in the audited source. The table is not an empirical superiority claim.

## Repository layout

```text
configs/          dataset/model/prompt/counterfactual/study/analysis manifests
src/faireval/     benchmark, adapters, providers, metrics, prompts, runner, audit, analysis, mitigation
tests/            metrics, prompt invariants, providers, config contracts, runner, analysis, statistics
scripts/          preflight, figure build, environment provenance, run audit, analysis build
docs/             blueprint, runbook, literature audit, novelty/reviewer/submission guards
paper/            anonymous LNCS manuscript scaffold and vector-PDF figures
results/          generated outputs only; large/raw/sensitive outputs are not committed
```

## Reproducibility path

See `docs/LOCAL_RUNBOOK.md` for Windows PowerShell and WSL commands. The intended order is:

1. install from `requirements.txt` and editable package;
2. configure API credentials only in the process environment;
3. run zero-call config/environment/test/paper preflight;
4. freeze all six deterministic dataset instances and hashes;
5. compile the immutable run plan;
6. inspect `execute-plan` in dry-run mode;
7. execute only with an exact code commit SHA;
8. audit the completed raw log against the immutable plan;
9. generate analysis artifacts from frozen results;
10. populate paper tables/figures only from those generated artifacts.

`write_environment_manifest.py` records package versions, config hashes, credential-presence booleans and endpoint fingerprints without writing raw secrets.

## Integrity rules

- No empirical number is invented or hand-entered into the final paper.
- Pilot and confirmatory outputs are labeled and separated.
- Invalid generations are retained as observable behavior.
- No prompt/model/dataset/K/hyperparameter is selected after viewing test outcomes and then presented as confirmatory.
- Tables and figures are generated from frozen result artifacts and record input hashes + analysis commit.
- Existing FairEval/publication overlap is documented; anonymous submission follows the ECIR double-blind policy.

## ECIR constraints

- Full-paper abstract registration: **21 September 2026**.
- Full-paper deadline: **5 October 2026**.
- Springer LNCS format.
- Maximum **12 pages plus references**; appendices count inside the 12-page limit.
- Double-blind review.

See `docs/ECIR2027_SUBMISSION_COMPLIANCE.md` before export.

## Current status

- scientific redesign: implemented
- four RQs + metrics + PAIR formulation: implemented
- six dataset configurations/adapters: implemented
- six-family model/provider layer: implemented; final Llama host still must be frozen before pilot
- controlled prompt/cue/order machinery: implemented + unit-tested
- immutable dataset/run-plan/execution provenance: implemented
- raw-run provenance audit: implemented
- RQ1/RQ2 scoring, pairing, bootstrap/permutation/Wilcoxon/Holm analysis pipeline: implemented
- professional methodology PDF figures + audited 16-row comparison table: implemented
- empirical result values: intentionally **not populated** until experiments run
- next execution milestone: validate raw dataset schemas/checksums, freeze the Llama serving endpoint, run the variance/invalidity pilot, freeze confirmatory N, then execute the locked matrix
