# FairEval ECIR 2027 — Literature Comparison Audit

**Status:** pre-submission evidence audit, 2026-09-15  
**Purpose:** support the compact related-work comparison table in the ECIR paper without turning it into a marketing table.

## Coding rule

This matrix records **study-design coverage**, not empirical superiority.

- **Y** = explicitly established by the audited paper/metadata.
- **P** = partial or adjacent coverage; it is not equivalent to the FairEval design criterion.
- **—** = not established by the evidence audited here. This does **not** claim that a feature is impossible or absent from every auxiliary experiment in the paper.

The public paper should use a red check for **Y**, a gray circle for **P**, and a black cross for **—**. The caption must state that symbols encode documented design coverage only.

### Comparison dimensions

| Code | Criterion used in this audit |
|---|---|
| Fair | Explicit fairness, bias, or identity-disparity evaluation |
| Pref | Recommendation/response consequence is grounded in utility or user preference rather than list change alone |
| Psy | **Measured psychometric** personality from a dataset-native instrument, not merely prompted OCEAN labels/adjectives |
| Pair | Paired/counterfactual or otherwise matched identity/context control |
| Prompt | Prompt wording, persona cue, or cue-form reliability is explicitly varied as an evaluation factor |
| Multi | More than one LLM/model family/configuration is compared |
| Mit | A fairness/bias mitigation is evaluated |
| Repeat | Independent repeated generations/stochasticity are explicitly part of the evaluation design |
| Invalid | Malformed/invalid model outputs are retained and analyzed as an outcome rather than silently discarded |

## Audited matrix

| Work | Fair | Pref | Psy | Pair | Prompt | Multi | Mit | Repeat | Invalid |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zhang et al., RecSys 2023 — FaiRLLM | Y | P | — | P | — | — | — | — | — |
| Hua et al., EACL 2024 — UP5 | Y | Y | — | Y | — | — | Y | — | — |
| Deldjoo, CoRR 2024 — FairEvalLLM | Y | P | — | Y | Y | — | — | — | — |
| Jiang et al., WWW 2024 — Item-side Fairness | Y | Y | — | — | — | — | Y | — | — |
| Xu et al., EMNLP Findings 2024 — Implicit Ranking Unfairness | Y | Y | — | P | — | P | Y | — | — |
| Sakib & Das, IEEE BigData 2024 — Challenging Fairness | Y | P | — | P | P | Y | Y | — | — |
| Mizrahi et al., TACL 2024 — Multi-Prompt Evaluation | — | — | — | — | Y | Y | — | — | — |
| Deldjoo & Di Noia, ACM TIST 2025 — CFaiRLLM | Y | Y | — | Y | P | P | P | — | — |
| Liu et al., Scientific Reports 2025 — Fairness Identification | Y | Y | — | — | — | Y | Y | — | — |
| Kantharuban et al., ACL Findings 2025 — Stereotype or Personalization? | Y | P | — | Y | Y | Y | — | — | — |
| Neplenbroek et al., EMNLP 2025 — Reading Between the Prompts | Y | P | — | Y | Y | P | Y | — | — |
| Lutz et al., EMNLP Findings 2025 — Prompt Makes the Person(a) | P | — | — | P | Y | Y | — | — | — |
| Hua et al., EMNLP 2025 — Flaw or Artifact? | — | — | — | — | Y | Y | — | — | — |
| Weeber et al., ACL 2026 — One Persona, Many Cues | P | P | — | P | Y | Y | — | — | — |
| Sah & Lian, ICCV Workshops 2025 — PerFairX | Y | P | P | P | P | Y | — | — | — |
| **FairEval (ECIR 2027 design)** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |

## Evidence ledger

### 1. FaiRLLM — Zhang et al., RecSys 2023

**Reference:** Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He. *Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation.* RecSys 2023, 993–999. DOI: `10.1145/3604915.3608860`.

**Evidence used:** the paper explicitly studies sensitive-attribute effects in ChatGPT recommendations across recommendation scenarios. This supports **Fair=Y**. We use **Pref=P** and **Pair=P** rather than claiming held-out preference-conditioned consequence estimation equivalent to FairEval.

### 2. UP5 — Hua et al., EACL 2024

**Reference:** Wenyue Hua et al. *UP5: Unbiased Foundation Model for Fairness-aware Recommendation.* EACL 2024, 1899–1912. DOI: `10.18653/v1/2024.eacl-long.114`.

**Evidence used:** fairness-aware recommendation, counterfactually fair prompting, recommendation effectiveness and mitigation are core parts of the work. Thus **Fair=Y, Pref=Y, Pair=Y, Mit=Y**.

### 3. FairEvalLLM — Deldjoo, 2024

**Reference:** Yashar Deldjoo. *FairEvalLLM. A Comprehensive Framework for Benchmarking Fairness in Large Language Model Recommender Systems.* CoRR abs/2405.02219 (2024). DOI: `10.48550/arXiv.2405.02219`.

**Evidence used:** the abstract explicitly describes multiple fairness dimensions, counterfactual evaluations, LastFM-1K and MovieLens-1M, and more than 50 prompt/profile/in-context scenarios over 80 users per dataset. Therefore **Fair=Y, Pair=Y, Prompt=Y**. Preference consequences are adjacent but not coded as the same held-out utility estimand used in FairEval, so **Pref=P**.

### 4. Item-side Fairness — Jiang et al., WWW 2024

**Reference:** Meng Jiang, Keqin Bao, Jizhi Zhang, Wenjie Wang, Zhengyi Yang, Fuli Feng, Xiangnan He. *Item-side Fairness of Large Language Model-based Recommendation System.* WWW 2024, 4717–4726. DOI: `10.1145/3589334.3648158`.

**Evidence used:** the paper explicitly evaluates item-side exposure/fairness and proposes IFairLRS, evaluated on MovieLens and Steam with recommendation quality/fairness reporting. Thus **Fair=Y, Pref=Y, Mit=Y**.

### 5. Implicit Ranking Unfairness — Xu et al., EMNLP Findings 2024

**Reference:** Chen Xu, Wenjie Wang, Yuxin Li, Liang Pang, Jun Xu, Tat-Seng Chua. *A Study of Implicit Ranking Unfairness in Large Language Models.* Findings of EMNLP 2024, 7957–7970. DOI: `10.18653/v1/2024.findings-emnlp.467`.

**Evidence used:** the paper studies discriminatory ranking from sensitive and non-sensitive profile cues and introduces fair-aware pairwise-regression data augmentation. It reports fairness and ranking-accuracy trade-offs. Thus **Fair=Y, Pref=Y, Mit=Y**; matched profile/name manipulation is coded conservatively as **Pair=P**.

### 6. Challenging Fairness — Sakib & Das, IEEE BigData 2024

**Reference:** Shahnewaz Karim Sakib, Anindya Bijoy Das. *Challenging Fairness: A Comprehensive Exploration of Bias in LLM-Based Recommendations.* IEEE BigData 2024, 1585–1592. DOI: `10.1109/BIGDATA62323.2024.10825082`.

**Evidence used:** the work evaluates bias across recommendation domains, demographic/cultural groups, different LLMs, intersectional/contextual factors, and prompt-engineering intervention. We therefore code **Fair=Y, Multi=Y, Mit=Y**, with matched/control and prompt-reliability dimensions as **P** because their purpose is not identical to FairEval's pre-registered robustness protocol.

### 7. Multi-Prompt Evaluation — Mizrahi et al., TACL 2024

**Reference:** Moran Mizrahi et al. *State of What Art? A Call for Multi-Prompt LLM Evaluation.* TACL 12 (2024), 933–949. DOI: `10.1162/tacl_a_00681`.

**Evidence used:** broad multi-model, multi-task prompt evaluation directly supports **Prompt=Y, Multi=Y**. It is methodological support rather than a recommender-fairness study.

### 8. CFaiRLLM — Deldjoo & Di Noia, ACM TIST 2025

**Reference:** Yashar Deldjoo, Tommaso Di Noia. *CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System.* ACM TIST 16(6), Article 142 (2025). DOI: `10.1145/3725853`.

**Evidence used:** the paper explicitly argues that list discrepancy alone can mislabel preference-aligned personalization as unfairness, uses true-preference alignment, sensitive-attribute conditions/intersections, and multiple profile-construction strategies. Hence **Fair=Y, Pref=Y, Pair=Y**; profile-construction sensitivity is **Prompt=P**, not the same as FairEval's wording/cue/order reliability audit.

### 9. Fairness Identification — Liu et al., Scientific Reports 2025

**Reference:** Wei Liu, Baisong Liu, Jiangcheng Qin, Xueyuan Zhang, Weiming Huang, Yangyang Wang. *Fairness identification of large language models in recommendation.* Scientific Reports 15, 5516 (2025). DOI: `10.1038/s41598-025-89965-3`.

**Evidence used:** the article uses MovieLens/LastFM and two LLM fairness recognizers (ChatGLM3-6B and Llama2-13B), reconstructs recommendations identified as unfair, and reports a fairness–utility trade-off. Thus **Fair=Y, Pref=Y, Multi=Y, Mit=Y**.

### 10. Stereotype or Personalization? — Kantharuban et al., ACL Findings 2025

**Reference:** Anjali Kantharuban, Jeremiah Milbauer, Maarten Sap, Emma Strubell, Graham Neubig. *Stereotype or Personalization? User Identity Biases Chatbot Recommendations.* Findings of ACL 2025, 24418–24436. DOI: `10.18653/v1/2025.findings-acl.1254`.

**Evidence used:** explicit and implicit racial identity cues affect recommendations across multiple consumer LLMs. The paper is directly about disentangling desired personalization from identity stereotyping. Hence **Fair=Y, Pair=Y, Prompt=Y, Multi=Y**. It does not use FairEval's held-out recommendation utility, so **Pref=P**.

### 11. Reading Between the Prompts — Neplenbroek et al., EMNLP 2025

**Reference:** Vera Neplenbroek, Arianna Bisazza, Raquel Fernández. *Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization.* EMNLP 2025, 20367–20400. DOI: `10.18653/v1/2025.emnlp-main.1029`.

**Evidence used:** controlled synthetic conversations probe demographic inference from stereotypical cues, including conflict with explicit identity; a representation-steering mitigation is evaluated. Hence **Fair=Y, Pair=Y, Prompt=Y, Mit=Y**. Response quality is adjacent to recommendation utility, so **Pref=P**.

### 12. Prompt Makes the Person(a) — Lutz et al., EMNLP Findings 2025

**Reference:** Marlene Lutz, Indira Sen, Georg Ahnert, Elisa Rogers, Markus Strohmaier. *The Prompt Makes the Person(a): A Systematic Evaluation of Sociodemographic Persona Prompting for Large Language Models.* Findings of EMNLP 2025, 23212–23237. DOI: `10.18653/v1/2025.findings-emnlp.1261`.

**Evidence used:** systematic persona-construction analysis over multiple LLMs and intersectional groups. Thus **Prompt=Y, Multi=Y**, while fairness relevance and matched group manipulation are coded **P** because this is not a recommender-fairness utility benchmark.

### 13. Flaw or Artifact? — Hua et al., EMNLP 2025

**Reference:** Andong Hua, Kenan Tang, Chenhe Gu, Jindong Gu, Eric Wong, Yao Qin. *Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs.* EMNLP 2025, 19889–19899. DOI: `10.18653/v1/2025.emnlp-main.1006`.

**Evidence used:** multi-model, multi-benchmark, multi-template analysis supports **Prompt=Y, Multi=Y** and is used to prevent FairEval from over-interpreting apparent prompt instability.

### 14. One Persona, Many Cues — Weeber et al., ACL 2026

**Reference:** Franziska Weeber, Vera Neplenbroek, Jan Batzner, Sebastian Padó. *One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization.* ACL 2026, 44892–44921. DOI: `10.18653/v1/2026.acl-long.2079`.

**Evidence used:** multiple sociodemographic cue forms, multiple LLMs and tasks demonstrate that cue realization changes personalization/bias conclusions. Thus **Prompt=Y, Multi=Y**; fairness/personalization consequence and matched-cue dimensions are conservatively **P**.

### 15. PerFairX — Sah & Lian, ICCV Workshops 2025

**Reference:** Chandan Kumar Sah, Xiaoli Lian. *PerFairX: Is There a Balance Between Fairness and Personality in Large Language Model Recommendations?* ICCV Workshops 2025, 2771–2780.

**Evidence used:** two LLMs (ChatGPT and DeepSeek), movie/music domains, OCEAN-conditioned personalization, and demographic-fairness analysis. Thus **Fair=Y, Multi=Y**. Because OCEAN context is not the same design as FairEval's dataset-native measured psychometric profile plus true-vs-shuffled negative control, **Psy=P**, not Y. Prompt, preference and paired-control dimensions are also coded **P** rather than overstated.

**Double-blind note:** this is prior author work. Keep it in the internal overlap/comparison audit. Before ECIR submission, apply the venue's self-citation/anonymity rule consistently; do not use wording that reveals authorship.

## Why FairEval's row is different

The intended novelty claim is the **joint design**, not that every individual component is unprecedented. Specifically, FairEval combines:

1. candidate-constrained held-out recommendation utility;
2. paired demographic interventions with unchanged preference evidence and candidate sets;
3. dataset-native **measured** Big Five profiles;
4. whole-profile derangement and one-trait observed-value negative controls;
5. pre-registered prompt, cue, candidate-order, K and repeated-generation robustness;
6. six datasets and six independent LLM families;
7. invalid-output disparity as an observable model behavior; and
8. mitigation evaluated on a utility–fairness–personalization frontier.

The paper must avoid the sentence “no prior work does X” unless a systematic search supports it. Preferred phrasing: **“To our knowledge, we are not aware of prior work that evaluates these dimensions jointly under a frozen, consequence-aware protocol.”**

## Integrity rules for the manuscript table

- Do not turn **P** into a red check to make FairEval look stronger.
- Do not mark a prior paper **—** merely because its abstract does not mention a feature if the full methods establish it; update this ledger first.
- Do not use citation counts as quality evidence.
- Do not compare unpublished FairEval numerical outcomes against published numbers until the frozen experiments finish.
- Keep `docs/PUBLICATION_OVERLAP_GUARD.md` synchronized with PerFairX and the prior IntechOpen chapter.
