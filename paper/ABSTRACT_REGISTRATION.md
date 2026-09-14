# ECIR 2027 Full-Paper Abstract Registration Draft

## Recommended anonymous submission title

**Separating Personalization from Discrimination in Personality-Aware LLM Recommendations**

## Abstract

Large language models (LLMs) increasingly act as conversational recommenders, yet evaluating their fairness is difficult because a recommendation that changes after receiving additional user context is not necessarily unfair: it may reflect useful personalization. Existing LLM-recommendation audits can conflate ranking sensitivity with discrimination, rely on weakly grounded personality descriptions, or evaluate too few models and domains to establish reliable conclusions. We propose a preference-conditioned evaluation framework that separates behavioral ranking change from its consequences for held-out user utility and item exposure. The benchmark combines real interaction histories with controlled demographic counterfactuals and measured Big Five personality profiles, including shuffled-personality and one-trait counterfactual controls. We evaluate six LLM families across six recommendation datasets spanning movies, music, short video, and news using candidate-constrained ranking and repeated generations. The study jointly measures recommendation effectiveness, counterfactual utility and exposure gaps, invalid-output disparities, personality-specific value, and cross-model/domain reliability. Finally, we compare prompting and preference-aligned counterfactual re-ranking strategies designed to reduce harmful identity-conditioned effects without erasing useful personalization. This design reframes fairness auditing from asking whether recommendations change to asking whether contextual changes are preference-justified, reliable, and equitable.

## Keywords

LLM recommendation; fairness; personalization; personality; Big Five; counterfactual evaluation; recommender systems

## Before submitting this registration

- Confirm final track: regular Full Paper versus IR-for-Good.
- Keep the anonymous title above rather than the already-public FairEval title.
- Do not add numerical claims until the experiments have actually run.
- Keep scope aligned with the full paper: six datasets, six model families, measured personality, counterfactual controls, reliability, and mitigation.
