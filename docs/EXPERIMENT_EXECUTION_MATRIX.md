# FairEval ECIR 2027 — Experiment Execution Matrix

This document converts the research design into a staged run plan. It deliberately avoids a full Cartesian product across every dataset, condition, prompt, cue, cutoff, candidate order, and generation seed. Such a design would spend most calls on nuisance-factor combinations and make the main estimands harder to interpret.

## Core principle

Run the scientifically necessary contrasts on the largest paired user samples. Run robustness factors **one at a time** on frozen subsets. Reuse unmitigated rankings for RQ4 and perform PAIR as post-processing rather than re-querying an LLM.

## Stage A — personality-grounded main study

Datasets: Personality 2018, Music Master BFI/BFI-2, REASONER.

Per dataset:

- 100 users;
- 6 model families;
- conditions C0 history-only, C3 true measured personality, C4 shuffled personality;
- 3 generations per cell.

Expected primary generations:

`3 datasets × 100 users × 6 models × 3 conditions × 3 repetitions = 16,200`

This stage answers the main RQ2 contrast without conflating personality with demographics.

## Stage B — one-trait personality intervention

Datasets: same three personality-grounded datasets.

Per dataset:

- frozen 30-user subset;
- 6 model families;
- five Big Five dimensions intervened one at a time;
- 3 generations per cell.

Expected generations:

`3 × 30 × 6 × 5 × 3 = 8,100`

This is a focused mechanistic analysis rather than a second full benchmark. Intervention values and direction must be frozen before the run and respect the scoring range of the source instrument.

## Stage C — demographic counterfactual main study

Datasets: MovieLens-1M and the frozen Last.fm release.

Per dataset:

- 100 users;
- 6 model families;
- C0 preference-only, C1 observed demographic context, C2 paired counterfactual context;
- 3 generations per cell.

Expected generations:

`2 × 100 × 6 × 3 × 3 = 10,800`

The primary attribute set must be frozen after dataset inspection but before LLM calls. Do not add an attribute because it produces a larger disparity.

## Stage D — MIND stress test

MIND does not provide observed demographic identity. Therefore this stage is explicitly synthetic and cannot support observed-group fairness claims.

- 50 users/impression tasks;
- 6 model families;
- preference-only vs one pre-registered synthetic counterfactual condition;
- 3 generations.

Expected generations:

`1 × 50 × 6 × 2 × 3 = 1,800`

## Core evidence total

Stages A–D require approximately **36,900 primary generations**, before format-repair overhead. Persistent invalid responses remain outcomes; they are not re-run until valid.

## Stage E — task-paraphrase robustness

Across all six datasets, use 20 frozen users per dataset, two representative conditions per dataset, six model families, three pre-registered task templates, and one generation per cell.

Expected generations:

`6 × 20 × 6 × 2 × 3 = 4,320`

This stage estimates task-wording variance. It is not used to choose the best prompt.

## Stage F — demographic-cue robustness

Only MovieLens-1M and Last.fm use the demographic cue suite in the main fairness analysis.

- 20 users per dataset;
- 6 model families;
- two representative demographic conditions;
- three cue forms;
- one generation.

Expected generations:

`2 × 20 × 6 × 2 × 3 = 1,440`

Cue forms carry the same attribute values and add no preference information.

## Stage G — candidate-order robustness

Across all six datasets:

- 20 users per dataset;
- 6 model families;
- two representative conditions;
- three deterministic permutations;
- one generation.

Expected generations:

`6 × 20 × 6 × 2 × 3 = 4,320`

Matched counterfactuals always share each permutation.

## Stage H — cutoff robustness

Across all six datasets:

- 20 users per dataset;
- 6 model families;
- two representative conditions;
- K in {5, 10, 20};
- one generation.

Expected generations:

`6 × 20 × 6 × 2 × 3 = 4,320`

## Stage I — focused stochasticity audit

Across all six datasets:

- 10 frozen users per dataset;
- 6 model families;
- two representative conditions;
- 10 generations at the pre-specified stochasticity setting.

Expected generations:

`6 × 10 × 6 × 2 × 10 = 7,200`

Provider seed support is logged; seeds are not assumed comparable across vendors.

## Robustness total

Stages E–I require approximately **21,600 generations**.

## Stage J — RQ4 instruction mitigation

Reuse all unmitigated Stage C results. On 50 users per demographic dataset:

- 6 model families;
- one additional identity-irrelevance prompt mode;
- 3 repetitions.

Expected additional API generations:

`2 × 50 × 6 × 1 × 3 = 1,800`

PAIR is then applied as deterministic post-processing to already generated counterfactual rankings and does not require new API calls.

## Planned total

Approximate planned LLM generations before format-repair overhead:

- core evidence: 36,900;
- robustness: 21,600;
- RQ4 additional prompt mitigation: 1,800;
- **total: 60,300 generations**.

This is a ceiling for the planned design, not a mandate to retry invalid responses. A single allowed format-only repair can add calls only when validation fails, and those repair calls must be reported separately.

## Priority order if execution time becomes constrained

The paper should never substitute breadth for validity. The minimum defensible ordering is:

**P0:** Stage A + Stage C = 27,000 generations. These support the central RQ1/RQ2 story.

**P1:** Stage B + Stage D + Stage J = 11,700 additional generations. These add mechanistic personality, cross-domain stress testing, and mitigation.

**P2:** Stages E–I = 21,600 robustness generations. These support RQ3 and should be preserved as far as possible because prior reviews explicitly challenged sampling and prompt stability.

If any stage is reduced, freeze the reduction before looking at its outcomes and report it transparently. Do not selectively stop models/datasets because their early results look uninteresting.

## Run gates

No main API run begins until all of the following pass:

1. dataset card and license check;
2. user split/candidate generation hash frozen;
3. prompt invariant tests passing;
4. primary demographic attributes frozen;
5. measured personality score mapping checked against the source instrument;
6. exact model/provider configuration frozen;
7. 5-user × 6-model smoke test passes schema validation;
8. analysis script can consume synthetic/mock JSONL end-to-end;
9. result tables/figures are generated from files, not manually typed values.

## Stop conditions

Stop and fix the pipeline rather than continuing the main run if:

- candidate membership violations indicate a provider/schema adapter bug;
- a counterfactual pair changes anything beyond the intended intervention;
- candidate order differs within a pair;
- provider aliases resolve to an unrecorded new backend during the run;
- personality score normalization is inconsistent across users within a dataset;
- a dataset license or redistribution restriction is unresolved.
