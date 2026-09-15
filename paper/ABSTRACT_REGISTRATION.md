# ECIR 2027 Full-Paper Abstract Registration Draft

## Submission title

**FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness**

> Double-blind note: this title preserves the requested FairEval identity and matches the public line of work. Because earlier related work is public, author inference may be easier even though ECIR permits preprints. If the team later prefers stronger anonymity, change only the submission title before abstract registration; do not change the underlying RQs or benchmark after seeing results.

## Abstract

Large language models (LLMs) increasingly act as conversational recommenders, yet a recommendation that changes after receiving identity or personality context is not necessarily unfair: the change may be useful personalization, harmless sensitivity, or a harmful disparity. Existing LLM-recommendation audits can conflate these cases, rely on weakly grounded personality descriptions, or draw conclusions from too few models, domains, prompts, and generations. We redesign FairEval around preference-conditioned evaluation. Real interaction histories and held-out relevance labels make the consequence of a ranking change measurable; measured Big Five profiles replace informal personality adjectives; shuffled-profile and one-trait counterfactuals provide falsifiable personality controls; and demographic interventions hold preference evidence, candidates, and prompt structure fixed. The real-world benchmark spans six recommendation datasets and six hosted/API LLM families. A complementary transparency track adds two local open-weight models, while FairSynth-360 provides a project-generated controlled sanity benchmark whose synthetic identity labels are independent of relevance by construction. The protocol uses candidate-constrained ranking, repeated generations, pre-registered prompt/cue/order robustness, explicit invalid-output accounting, and utility--fairness analysis. We additionally evaluate identity-irrelevance prompting and contextual preference-aligned counterfactual re-ranking designed to reduce harmful identity-conditioned gaps without erasing useful personalization. FairEval therefore reframes auditing from asking only whether recommendations change to testing whether contextual changes are preference-justified, consequential, robust, and equitable.

## Keywords

LLM recommendation; fairness; personalization; personality; Big Five; counterfactual evaluation; recommender systems; robustness

## Registration checklist

- Target: ECIR 2027 regular Full Paper unless the final framing is deliberately moved to IR-for-Good.
- Abstract registration deadline: 21 September 2026, 23:59 GMT.
- Full-paper deadline: 5 October 2026, 23:59 GMT.
- Keep the abstract free of numerical findings until experiments generate them.
- Registered real-world scope: six datasets and six hosted/API model families.
- Registered extensions: two local open-weight models plus separately reported FairSynth-360 controlled sanity analysis.
- Never describe FairSynth synthetic OCEAN as measured human psychometrics or its A/B/C labels as demographic categories.
- Keep hosted-vs-local and real-vs-synthetic results stratified rather than pooling them into causal claims.
- Do not expand the scope after observing preliminary results merely to chase a stronger effect.
