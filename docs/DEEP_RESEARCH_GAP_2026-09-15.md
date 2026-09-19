# FairEval ECIR 2027 — Deep Research Gap (freeze candidate 2026-09-15)

## Executive position

The old FairEval story cannot be resubmitted as "recommendation lists change when identity/personality is mentioned, therefore the LLM is unfair." That premise is no longer defensible. CFaiRLLM already argues that list changes can reflect genuine preference-aligned personalization, and recent ACL/EMNLP work shows that identity cues, stereotypes, implicit personalization, and prompt formulation can all affect outputs.

The ECIR 2027 version should therefore ask a harder question:

> When an LLM recommendation changes after demographic or measured personality context is supplied, can we determine whether the change is useful personalization, harmless sensitivity, or a harmful identity-conditioned disparity — and are those conclusions reliable across models, domains, prompts, candidate order, and repeated generations?

That question creates a substantially different study from the 2025 FairEval paper.

---

## 1. What prior work already owns

### FaiRLLM / Zhang et al. (RecSys 2023)

Already studies sensitive-attribute effects in LLM recommendation using neutral/sensitive prompt comparisons. We must not present this basic perturbation design as new.

### UP5 / Hua et al. (EACL 2024)

Already studies fairness-aware foundation-model recommendation and counterfactually fair prompting. A prompt-only mitigation is therefore a baseline, not our methodological novelty.

### CFaiRLLM / Deldjoo & Di Noia (TIST 2025)

Already criticizes the assumption that recommendation-list differences automatically imply unfairness. It incorporates true preference alignment, intersectional prompts, and user-profile sampling strategies.

**Consequence:** "separating personalization from bias" by itself is not enough novelty for our paper.

### Kantharuban et al. (ACL Findings 2025)

Shows that chatbot recommendations can reflect both user wants and racial identity, including stereotypical recommendation effects under explicit and implicit identity disclosure.

### Neplenbroek et al. (EMNLP 2025)

Shows implicit demographic inference from conversational cues and stereotype-driven personalization, including quality degradation for some groups.

### Lutz et al. (EMNLP Findings 2025)

Shows that sociodemographic persona-prompt construction itself materially affects LLM simulations and stereotyping.

### Weeber et al. (ACL 2026)

Compares six sociodemographic cue types across seven LLMs and shows that cue choice can alter conclusions about persona-induced differences and bias. This makes single-cue fairness claims especially fragile in 2026.

### Multi-prompt / prompt-sensitivity evaluation

Mizrahi et al. (TACL 2024) demonstrates brittleness of single-prompt LLM evaluation. Hua et al. (EMNLP 2025) cautions that some apparent prompt sensitivity can be an evaluation artifact. Liu & Chu (ACL 2026) provide further evidence that meaning-preserving prompt variants can materially affect model outputs.

**Consequence:** prompt variance must be measured, but prompt sensitivity should not automatically be equated with unfairness.

---

## 2. Defensible FairEval novelty

The paper becomes strong only if the contributions are the combination below, not any one element alone.

### Contribution A — psychometrically grounded personality fairness

Replace informal adjectives and demographic-plus-occupation pseudo-personas with **measured Big Five instruments** from real-user recommendation datasets.

Primary personality resources:

1. GroupLens Personality 2018 (movies);
2. Music Master BFI/BFI-2 (music);
3. REASONER CBF-PI-15 responses (short video).

The key negative control is true measured personality vs a shuffled profile from another user. If both change rankings similarly, the result is not evidence that the LLM extracted useful user-specific personality information.

### Contribution B — behavioral change vs consequence decomposition

Do not define fairness from list similarity alone.

For each matched intervention report both:

- **behavioral shift**: RBO/Jaccard/rank change;
- **consequence**: held-out nDCG/Recall utility change and, where item metadata permits, exposure change.

This yields interpretable quadrants:

1. low shift / low consequence — stable;
2. high shift / utility gain — potentially useful personalization;
3. high shift / no material utility change — sensitivity without demonstrated harm;
4. high shift / utility loss or group disparity — harmful candidate for fairness concern.

The labels remain empirical/operational; they are not claims of legal discrimination.

### Contribution C — reliability as a first-class fairness question

Evaluate six independent model families across six datasets and explicitly decompose variation caused by:

- model family;
- domain/dataset;
- prompt paraphrase;
- demographic cue representation on a robustness subset;
- candidate order;
- K;
- repeated generation.

The goal is not to declare a universal "fairest model." The goal is to ask whether a fairness conclusion survives reasonable experimental perturbations.

### Contribution D — mitigation that preserves useful personalization

Compare:

1. unmitigated audit prompt;
2. counterfactually/fairness-aware prompting baseline;
3. FairEval PAIR re-ranking.

A successful mitigation must reduce harmful counterfactual utility/exposure gaps **without simply making every user receive the same list** and without destroying true-vs-shuffled personality value.

This preservation criterion is important: erasing all personalization is not automatically a good fairness outcome.

---

## 3. Four RQs — recommended freeze

### RQ1 — Consequence of identity-conditioned change

**Holding observed preferences, candidate items, task, and prompt structure fixed, when demographic context changes an LLM ranking, does the change alter held-out recommendation utility or exposure in a way consistent with useful personalization, neutral sensitivity, or harmful disparity?**

Primary evidence: paired CUG, exposure gap where supported, RBO/Jaccard as behavioral diagnostics only.

### RQ2 — Grounded personality value

**Does measured Big Five personality provide user-specific recommendation value beyond observed preference history, and does that value survive shuffled-profile and one-trait counterfactual controls?**

Primary evidence: Personality Value Added (true vs shuffled), nDCG/Recall, trait/facet analyses where instrument support is comparable.

### RQ3 — Reliability and generalization

**How stable are FairEval conclusions across six model families, six datasets, prompt/cue formulations, candidate order, ranking cutoff, and repeated generations, and which sources contribute most to observed variance?**

Primary evidence: variance components/mixed effects where identifiable, paired confidence intervals, conclusion-consistency rates, prompt/order sensitivity.

### RQ4 — Personalization-preserving mitigation

**Can lightweight instruction-based and counterfactual re-ranking interventions reduce harmful identity-conditioned gaps while preserving overall utility and beneficial personality-driven personalization?**

Primary evidence: utility–fairness Pareto frontier, delta-CUG/CEG, delta-nDCG, delta-PVA.

---

## 4. Six-dataset architecture

### Personality-grounded track

1. **Personality 2018 / GroupLens** — movie domain, real recommender users and measured personality.
2. **Music Master BFI/BFI-2** — music domain, 279 participants, BFI/BFI-2 facets, 5,278 rating entries, 745-song pool reported by the paper.
3. **REASONER** — short-video domain, real-user interaction/feedback data and CBF-PI-15 responses.

### Counterfactual/generalization track

4. **MovieLens 1M** — movie ratings with voluntarily supplied demographic profile fields; pin 1M, not `ml-latest`.
5. **Last.fm** — freeze one release before the pilot. For deadline practicality, prefer Last.fm-1K unless a reproducible LFM-1b preprocessing path is already available. Preserve upstream non-commercial/license constraints.
6. **MIND** — news clicks and logged impressions. No observed demographic/personality claims; identity conditions on MIND are synthetic stress tests only.

### Important dataset rule

Never infer protected/sensitive identity from user names, text, ZIP codes, embeddings, or model guesses. Distinguish **observed**, **derived**, and **counterfactual** fields in every dataset card.

---

## 5. Six-family model panel

Design-freeze candidates verified from official provider documentation on 2026-09-15:

1. OpenAI — GPT-5.6 Terra;
2. Anthropic — Claude Sonnet 5;
3. Google — Gemini 3.8 Flash;
4. DeepSeek — DeepSeek V4 Flash;
5. Alibaba/Qwen — Qwen 3.8 Max;
6. Meta — Llama 4 Maverick, with exact hosting provider/checkpoint still to be frozen.

### Interpretation rule

This is a family-diversity panel, **not a controlled model-size leaderboard**. Do not conclude that differences are caused solely by vendor/family because model size, post-training, safety policies, and serving stacks also differ.

---

## 6. Strong prompt design

The prompt protocol is separately frozen in `docs/PROMPT_PROTOCOL.md` and `configs/prompt_suite.yaml`.

Core rules:

- audit prompts contain no fairness coaching;
- RQ4 mitigation prompts are separate;
- personality values come from measured instruments, not adjectives;
- paired interventions keep history/candidates/order/template fixed;
- exact candidate IDs make evaluation objective;
- multiple semantic task templates quantify prompt variance;
- deterministic candidate permutations quantify order effects;
- raw outputs and invalid rates are retained;
- no chain-of-thought is requested;
- prompt hashes and model versions are logged.

---

## 7. Metrics hierarchy

### Primary utility

- nDCG@10;
- Recall@10.

### Behavioral diagnostics (not fairness by themselves)

- RBO@K;
- Jaccard@K;
- rank displacement.

### Fairness/consequence quantities

- Counterfactual Utility Gap (CUG);
- absolute CUG and signed paired distribution;
- Group Utility Disparity (GUD), where group interpretation is valid;
- Counterfactual Exposure Gap (CEG), only where reliable item-group metadata exists;
- Invalid Output Disparity (IOD).

### Personality quantity

- Personality Value Added (PVA): utility with true measured profile minus matched shuffled-profile utility.

### Legacy metric

PAFS remains a backward-compatible diagnostic only. It is not the headline fairness metric because the previous formulation lacks sufficient theoretical grounding and conflates consistency with fairness.

---

## 8. Statistical discipline

Pre-specify before the main run:

- user as primary unit of inference;
- paired analyses for counterfactuals;
- 95% paired bootstrap confidence intervals;
- paired permutation and/or Wilcoxon tests as appropriate;
- Holm correction within RQ families;
- effect sizes, not p-values alone;
- variance-component or mixed-effects analysis for RQ3 where the design supports it;
- explicit missing/invalid-output analysis;
- no replacing failed generations after seeing outcomes.

A result should not be called robust merely because an aggregate mean is similar. Report uncertainty over users and over repeated model generations.

---

## 9. Kill criteria / claims we must refuse to make

Do **not** write any of the following unless experiment evidence directly supports it:

- "A changed recommendation is unfair."
- "Personality improves fairness" without true-vs-shuffled controls.
- "Model X is the fairest LLM" from this heterogeneous six-family panel.
- "Demographic effect equals discrimination."
- "Prompt robustness is solved" from three templates.
- "The benchmark represents all cultures or identities."
- "MIND has demographic fairness labels."
- "PAFS proves fairness."

Do not fabricate API seeds, exact backend versions, demographic labels, or human-study validation.

---

## 10. ECIR fit

The project fits ECIR's explicit interest in recommender systems, algorithmic bias/fairness, trustworthy IR, robustness, and societally motivated IR. The full-paper limit is 12 pages plus unlimited references; appendices count toward the 12-page limit. This forces the main paper to be self-contained and prevents moving core results to an appendix — directly addressing prior reviewer criticism.

---

## 11. What would make this paper genuinely stronger than the rejected version

The strongest story is not "we added four more datasets and four more LLMs." Scale alone does not repair conceptual validity.

The stronger story is:

1. real preference evidence replaces open-ended recommendation guessing;
2. measured psychometrics replace pseudo-personality adjectives;
3. true-vs-shuffled personality provides a falsifiable control;
4. ranking change is separated from utility/exposure consequence;
5. cue/template/order/generation variance is measured rather than ignored;
6. mitigation is judged by whether it reduces harmful gaps **and preserves legitimate personalization**;
7. every result is generated from frozen code/manifests, with no hand-entered paper numbers.

That combination directly targets the weaknesses raised in the previous RecSys/AAAI-style reviews while moving beyond what CFaiRLLM and later persona-bias work already established.
