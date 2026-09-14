# Reviewer → Design Traceability Matrix

This file turns the prior rejection feedback into falsifiable design requirements. A row is not “resolved” because the manuscript promises to discuss it; it is resolved only when the implementation/results provide the listed evidence.

| Prior criticism | ECIR design response | Required evidence before paper freeze |
|---|---|---|
| Recommendation change was treated too directly as unfairness | Separate behavioral ranking shift from held-out utility/exposure consequence; counterfactual pairs keep preference evidence and candidates fixed | RQ1 table reports both shift metric and utility/exposure gap with paired CIs |
| Valid personalization was not distinguished from discrimination | Introduce preference-conditioned interpretation: beneficial change may be personalization; identity-driven quality/exposure harm is the fairness signal | Conceptual Fig. 1 + paired RQ1 analysis + stereotype/error case study |
| Personality was ad hoc / not a psychological construct | Use measured Big Five/OCEAN or BFI-2 data on personality-grounded datasets | Dataset cards show instrument/fields; prompts use numeric trait vectors |
| Personality may merely encode demographic/occupation conditions | Separate demographics and personality fields; one-factor counterfactual edits; never define personality using occupation | Prompt invariant tests + condition manifest |
| Artificial prompts lacked realism | Build prompts from real interaction histories and logged candidate sets; keep syntax controlled | Dataset adapters and prompt examples derived from real histories |
| Prompt wording introduced confounds | Field-based templates; same order/task/candidate list; neutral fields use explicit `unspecified`; multiple pre-registered templates | Prompt diff/invariant tests and template-variance analysis |
| Single-pass LLM sampling | Three repetitions in main matrix + ten repetitions on stratified stochasticity subset | Run manifest and RQ3 variance analysis |
| Generation parameters missing | Persist provider/model ID, timestamp, temperature/top-p/thinking/seed support, prompt hash, request hash, code commit | Machine-readable manifest shipped with results |
| Incomplete/malformed outputs were discarded | Strict schema; one format-only repair; persistent failures retained as outcome | Invalid Output Disparity metric + failure counts by condition/model |
| Only two LLMs | Six independent model families | Reproducibility table with exact six model snapshots |
| Only movie and music domains | Six datasets including movie, music, short video, and news | Dataset table + cross-domain heterogeneity analysis |
| No recommendation-accuracy analysis | Candidate-constrained held-out relevance evaluation with nDCG/Recall | Utility metrics reported beside every fairness conclusion |
| Accuracy–fairness trade-off absent | RQ4 utility–fairness Pareto evaluation | Pareto plot + validation protocol for mitigation strength |
| PAFS lacked theory / alternatives | Demote original PAFS to legacy sensitivity baseline; primary metrics are interpretable paired utility/exposure effects | Main results do not rely on PAFS for fairness claims |
| Metric terminology SNSR/SNSV unclear | Remove nonessential acronyms from primary story; define every retained diagnostic once | Symbol/metric table and unit tests |
| Framework novelty perceived incremental relative to FaiRLLM | Novelty rests on measured personality, preference-conditioned counterfactual diagnosis, negative controls, six-family/six-dataset reliability, repeated sampling, and mitigation evaluation | Related-work comparison table explicitly includes FaiRLLM, CFaiRLLM, UP5, FairEval-old, FairEval-new |
| Prior robustness experiment resembled prior work | Reliability now includes stochasticity, model-family heterogeneity, dataset/domain heterogeneity, prompt templates, K sensitivity, invalid-output disparity | RQ3 multi-source variance analysis |
| Sensitive attributes in user prompts seemed practically unmotivated | Treat explicit identity insertion as a controlled causal stress test, not as an assumption that ordinary users always state identity | Problem formulation distinguishes observed context, disclosed context, and counterfactual intervention |
| Main results were buried in appendix | Keep all primary RQ figures/tables in 12-page main text | Paper checklist verifies every headline claim points to a main-text figure/table |
| Formal notation was imprecise / deterministic despite stochastic model | Define ranking model as stochastic conditional distribution over rankings; define candidate universe, context, intervention, and repeated samples | Problem-formulation section + notation consistency test/manual review |
| “Advantaged/disadvantaged” labels could be unjustified | Prefer descriptive group names and paired gaps; use advantaged/disadvantaged only when normatively and empirically justified | Terminology audit before submission |
| Selection bias in manually curated artists/directors | Replace hand-curated open-ended target lists with established user-item datasets and explicit sampling protocol | Split/sampling scripts + dataset cards |
| Interpretability weak | Provide paired counterfactual examples with preference history, list change, utility change, and category/exposure decomposition | RQ1 qualitative error taxonomy sampled by pre-specified rules |
| No mitigation | Evaluate unmitigated, identity-irrelevance prompting, prior counterfactual prompting baseline, and PAIR reranking | RQ4 comparison + ablations |
| Results resource-heavy | Separate pilot/main/stochasticity tiers; cache immutable prompts/responses; publish a small reproducibility subset where licenses permit | Cost/runtime report and reproduction command |

## Definition of “done”

A prior concern is considered addressed only if:

1. the repository contains executable support for the response;
2. the experiment manifest records the necessary variables;
3. a main-text analysis reports the corresponding evidence;
4. limitations are stated where the benchmark cannot resolve the concern.
