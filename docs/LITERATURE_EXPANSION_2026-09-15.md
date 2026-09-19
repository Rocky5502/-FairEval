# FairEval ECIR 2027 — Literature Expansion Ledger (2026-09-15)

This file records literature added after the main comparison-table audit. The goal is to strengthen motivation and methodological grounding without inflating the comparison table with papers that answer adjacent rather than identical questions.

## A. Domain-specific LLM recommendation fairness

### FairWork — SIGIR 2025
**Yuhan Hu, Ziyu Lyu, Lu Bai, Lixin Cui.** *FairWork: A Generic Framework For Evaluating Fairness In LLM-Based Job Recommender System.* SIGIR 2025, pp. 3964–3968. DOI `10.1145/3726302.3730145`.

Verified against the official SIGIR 2025 program/DOI metadata. FairWork evaluates sensitive-attribute effects in LLM job recommendation from both user and recruiter perspectives and uses an explicitly fairness-oriented evaluation workflow. Its framework perturbs sensitive attributes while keeping job/candidate evidence fixed, so paired/counterfactual evaluation is explicit. It is strong evidence that domain-specific, stakeholder-aware LLM recommendation auditing is becoming important. It does **not** establish FairEval's measured-personality controls, repeated-generation protocol, or invalid-output accounting.

**Comparison-table coding:** `Fair=Y`, `Pref=P`, `Psy=—`, `Pair=Y`, `Prompt=—`, `Multi=P`, `Mit=—`, `Repeat=—`, `Invalid=—`. `Pref=P` is deliberately conservative because the available evidence establishes qualification/relevance-aware job matching and fairness metrics, not FairEval's held-out preference-conditioned recommendation-utility estimand. `Multi=P` records adjacent evidence of multiple LLM configurations without treating model breadth as the central evaluated contribution.

**Use in FairEval:** related-work narrative and motivation for consequence/stakeholder-aware fairness. Do not code it as evidence for psychometric grounding.

## B. Personality-aware conversational recommendation

### PPPG-DialoGPT — IEEE WI-IAT 2023
**Fahed Elourajini, Esma Aïmeur.** *PPPG-DialoGPT: A Prompt-based and Personality-aware Framework For Conversational Recommendation Systems.* WI-IAT 2023, pp. 273–279. DOI `10.1109/WI-IAT59888.2023.00044`.

Verified against the IEEE/WI-IAT program and DBLP metadata. The work combines personality and prompt-based modeling in a conversational movie-recommendation setting and reports improved recommendation/response generation performance on TG-ReDial.

**Use in FairEval:** establishes that personality-conditioned conversational recommendation predates the present study. It should not be described as using FairEval's dataset-native measured Big Five negative controls unless the full paper explicitly supports that stronger claim.

## C. Psychometric validity and personality expression in LLMs

### PersonaLLM — Findings of NAACL 2024
**Hang Jiang, Xiajie Zhang, Xubo Cao, Cynthia Breazeal, Deb Roy, Jad Kabbara.** *PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits.* Findings of NAACL 2024. DOI `10.18653/v1/2024.findings-naacl.229`.

The paper evaluates GPT-3.5/GPT-4 personas using the Big Five framework and BFI-based assessment, showing why personality prompting should be evaluated rather than assumed to instantiate a stable human-like trait profile.

### LLMs Simulate Big5 Personality Traits — PERSONALIZE 2024
**Aleksandra Sorokovikova, Sharwin Rezagholi, Natalia Fedorova, Ivan P. Yamshchikov.** *LLMs Simulate Big5 Personality Traits: Further Evidence.* PERSONALIZE 2024, pp. 83–87. DOI `10.18653/v1/2024.personalize-1.7`.

This work studies whether several LLMs can simulate Big Five traits and their stability. It supports FairEval's decision to distinguish an LLM's simulated persona from a user's **measured** psychometric profile.

### BIG5-CHAT — ACL 2025
**Wenkai Li, Jiarui Liu, Andy Liu, Xuhui Zhou, Mona T. Diab, Maarten Sap.** *BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data.* ACL 2025, pp. 20434–20471. DOI `10.18653/v1/2025.acl-long.999`.

BIG5-CHAT explicitly critiques purely descriptive prompting as a weak way to instantiate personality and introduces human-grounded personality data. This is strong methodological support for FairEval's insistence on dataset-native measured psychometrics instead of free-form adjectives.

### Persona-E2 — ACL 2026
**Yuqin Yang et al.** *Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events.* ACL 2026, pp. 29290–29315. DOI `10.18653/v1/2026.acl-long.1350`.

Persona-E2 provides human-grounded personality labels and reports that surface role-play can suffer from a "personality illusion." Although not a recommender-system paper, it reinforces the need for externally grounded personality evidence and falsifiable controls.

## Manuscript positioning enabled by this pass

The strengthened claim should be:

> Personality-conditioned generation and recommendation are established research directions, but simulated persona expression is not equivalent to measured user psychometrics. FairEval therefore uses dataset-native Big Five measurements and tests whether those measurements add user-specific recommendation value against shuffled-profile and one-trait controls.

Avoid stronger statements such as "prior work does not use personality" or "prompted personality is invalid." The evidence supports a distinction between **prompted/simulated persona** and **externally measured psychometric user context**, not a blanket rejection of persona prompting.

## Citation-integrity rule

All entries in `paper/references_extra.bib` must be traceable to publisher/ACL/IEEE/SIGIR metadata or another primary bibliographic source. If a metadata field cannot be verified, leave it out rather than infer it.
