# FairEval ECIR 2027 Prompt Protocol

## Purpose

Prompting is part of the experimental design, not a tuning trick. The protocol is frozen before the main model run so prompt variants cannot be selected after looking at fairness or utility outcomes.

## 1. Two strictly separated system modes

### Audit mode (RQ1--RQ3)

The primary audit system message is deliberately neutral. It asks the model to rank only supplied candidate IDs and obey the output schema. It must not mention fairness, stereotypes, protected groups, debiasing, or mitigation. Otherwise the baseline would already contain an intervention.

### Identity-irrelevance mode (RQ4 only)

A separate mitigation system message tells the model to prioritize observed preference evidence and not treat demographic context as a proxy for unstated preference. This condition is never mixed into RQ1--RQ3.

This separation directly prevents a common confound: comparing a debiased system against itself and calling the difference a fairness measurement.

## 2. Primary prompt representation

The primary prompt is structured and candidate-constrained. Every prompt contains the same ordered sections:

1. task identifier and task instruction;
2. dataset identifier;
3. observed preference history;
4. user context block;
5. candidate items;
6. output contract.

Absent context is represented explicitly as `unspecified`; sections are not deleted. A demographic or personality intervention is allowed to change only its designated context values.

The model must return exactly K unique candidate IDs in rank order. No free-form explanation or chain-of-thought is requested.

## 3. Personality representation

Personality is never synthesized from informal labels such as "introverted engineer". For personality-grounded datasets, the prompt receives measured scores from the dataset-native psychometric instrument. The instrument name and score view are stored in metadata.

Primary analysis uses numeric OCEAN/Big-Five values. We do not infer personality from names, text, demographics, or embeddings.

Controls:

- preference history only;
- true measured personality;
- personality profile shuffled across users under a frozen seed;
- one-trait-at-a-time counterfactual while the other measured dimensions remain fixed.

True-vs-shuffled personality is the key negative control: if both produce similar changes, merely adding personality-shaped information is not evidence of useful personalization.

## 4. Counterfactual invariants

For every matched comparison, all of the following are identical unless they are the pre-registered intervention:

- user and held-out labels;
- preference history and history order;
- candidate set;
- candidate order;
- K;
- task wording/template ID;
- system mode;
- decoding configuration;
- model snapshot/provider route;
- output schema.

The repository's prompt parser is used in tests to verify which fields changed. A demographic intervention must not silently change personality, candidates, preferences, or task wording; a personality intervention must not silently change demographics.

## 5. Candidate-order effects

Closed-set ranking reduces hallucination but candidate position can affect LLM rankings. Therefore:

- the primary comparison uses one deterministic candidate ordering per user and reuses it across every matched condition;
- order robustness is a separate secondary analysis using pre-registered deterministic permutations;
- we never resample candidate order independently across a counterfactual pair;
- the order seed/permutation ID is logged with every request.

Primary claims cannot depend on choosing the most favorable candidate permutation.

## 6. Prompt-template robustness

The code defines three semantically matched task instructions:

- `field_v2_a` -- primary;
- `field_v2_b` -- paraphrase robustness;
- `field_v2_c` -- paraphrase robustness.

All share identical JSON keys, values, field order, and output contract. The main paper reports `field_v2_a`; RQ3 estimates template variance across all three on a pre-registered robustness subset.

We do **not** select the best-performing template after observing results. If the three templates disagree materially, that disagreement is a result and a limitation.

## 7. Natural-language external-validity subset

The primary benchmark prioritizes causal control over conversational realism. To address external validity without contaminating the main study, a smaller pre-registered subset may use a natural-language rendering of exactly the same evidence and candidates.

Rules:

- no new preference facts may be added;
- no demographic or personality descriptors may be paraphrased into stereotypes;
- candidate IDs and metadata remain identical;
- held-out labels are never shown;
- the subset is analyzed separately and cannot replace the primary structured result.

If time does not permit a defensible validation of this subset, it is omitted rather than improvised.

## 8. Output validation

A response is valid only when it contains exactly K unique IDs and every ID belongs to the supplied candidate set. Raw responses are retained.

At most one format-only repair may be attempted. A repair may serialize existing choices but may not add, delete, replace, or reorder item choices. Persistent failures remain outcomes and contribute to invalid-output disparity rather than being silently dropped.

## 9. Prompt/version provenance

Every run stores:

- prompt template ID;
- system mode;
- prompt SHA-256;
- candidate-order permutation ID or seed;
- requested and resolved model version;
- decoding parameters;
- provider seed support;
- request/response hash;
- repetition number;
- code commit SHA;
- timestamp.

This permits exact prompt auditing even when proprietary API model aliases later change.

## 10. Main and robustness allocation

### Main factorial cells

Use:

- template: `field_v2_a`;
- audit mode for RQ1--RQ3;
- one frozen candidate order per user;
- K=10;
- three independent generations per cell.

### Focused robustness subset

On a frozen stratified subset, cross:

- three task paraphrases;
- three candidate-order permutations;
- K in {5, 10, 20};
- ten generations for stochasticity estimation.

This robustness suite is intentionally concentrated on a subset so the six-family/six-dataset study remains computationally feasible.

## 11. RQ4 prompt baselines

RQ4 compares at least:

1. unmitigated audit prompt;
2. identity-irrelevance instruction;
3. preference-aligned counterfactual re-ranking (PAIR).

All three use the same user instances, candidates, held-out outcomes and evaluation metrics. We report the utility--fairness Pareto behavior rather than selecting a post-hoc operating point.

## 12. Reporting rules

The paper must report prompt text or a complete machine-readable equivalent, decoding parameters, invalid-output rates, number of generations, prompt-template allocation, candidate-order protocol, and whether provider seeds were actually supported.

Behavioral list change (Jaccard/RBO) is reported as sensitivity, not automatically labeled unfairness. Fairness conclusions must be supported by consequence-aware quantities such as held-out utility/exposure changes under matched interventions.
