# FairEval ECIR 2027 — Research Blueprint

## 0. Submission target and design constraint

Target: **ECIR 2027 Full Paper**, LNCS format, maximum 12 pages excluding references (appendices count toward the 12-page limit). Because the problem is societally motivated and directly studies algorithmic fairness, the ECIR **IR-for-Good** core track is also a strong strategic fit. The final track choice should be made before submission; the experimental design below works for either track.

Working title:

> **FairEval: Separating Personalization from Discrimination in Personality-Aware LLM Recommendations**

The title intentionally preserves the FairEval identity without exactly duplicating the existing arXiv/report title.

---

## 1. Why this must be a new study rather than an incremental extension

The earlier FairEval manuscript had a promising problem but three intertwined validity failures:

1. **List change was too close to the fairness definition.** A recommendation can legitimately change when new information is genuinely useful to the user's preference. Therefore list divergence is evidence of *sensitivity*, not by itself evidence of unfairness.
2. **Personality was insufficiently grounded.** Personality must be represented by validated psychometric measurements (Big Five/OCEAN or BFI-2 facets), not informal adjectives or demographic+occupation composites.
3. **The evaluation lacked enough causal/reliability controls.** Single generations, only two model families/domains, synthetic prompt construction, missing utility analysis, and limited mitigation make causal interpretation weak.

The 2025 CFaiRLLM paper independently makes a closely related criticism of similarity-only fairness audits: recommendation differences can reflect true preference-aligned personalization. Consequently, simply adding preference utility to the old FairEval would no longer be a sufficient novelty claim.

### New scientific thesis

> **Fairness in LLM recommendation should be evaluated conditionally on user preference evidence. Personality is useful context only when it provides measured, preference-aligned value; demographic identity should not induce quality or exposure disparities when preferences and candidate items are held constant.**

The study therefore moves from a one-dimensional “did the list change?” question to a two-axis diagnosis:

- **behavioral effect:** how much did the ranking change?
- **preference-conditioned consequence:** did the change improve utility, preserve utility, or create a systematic quality/exposure loss?

This yields interpretable cases:

| Ranking changes? | Utility consequence | Interpretation |
|---|---|---|
| No | stable | insensitive / stable |
| Yes | improves for true preference/personality evidence | legitimate personalization candidate |
| Yes | no utility gain, identity-correlated | suspicious identity sensitivity |
| Yes | utility/exposure worsens for a counterfactual identity | discriminatory-effect candidate |

The paper should avoid claiming that observational experiments establish legal discrimination. We measure *counterfactual disparities and stereotype-consistent effects* under controlled benchmark conditions.

---

## 2. Four research questions

### RQ1 — Personalization vs. discrimination

**When demographic or personality context changes an LLM-generated ranking, when are those changes preference-aligned personalization and when do they produce unjustified counterfactual utility or exposure gaps after holding the user's preference history, task, candidates, and prompt syntax fixed?**

Primary tests:
- neutral vs observed identity where available;
- neutral vs counterfactual identity;
- identity A vs identity A' paired within the same user/candidate context;
- change in rank similarity **and** held-out utility;
- exposure/category changes when item metadata permits.

### RQ2 — Grounded personality value

**Do measured Big Five traits add recommendation value beyond observed preference history, and are those gains specific to the true personality profile rather than prompt perturbation or stereotypical shortcuts?**

Controls:
- preference-only baseline;
- true measured Big Five profile;
- shuffled personality profile from another user;
- one-trait counterfactual edit while other OCEAN dimensions are fixed;
- trait-stratified performance and calibration.

A personality claim is supported only when true profiles outperform shuffled/null controls in a reproducible way.

### RQ3 — Generalization and reliability

**How stable are personalization/fairness conclusions across six datasets, six LLM families, prompt templates, ranking cutoffs, and repeated generations?**

We explicitly estimate:
- between-model heterogeneity;
- between-dataset/domain heterogeneity;
- prompt-template variance;
- generation stochasticity;
- sensitivity to K;
- invalid-output disparity.

### RQ4 — Mitigation without erasing useful personalization

**Can low-cost prompting and preference-aligned counterfactual re-ranking reduce harmful identity-conditioned gaps while preserving recommendation utility and genuine personality gains?**

Baselines:
1. unmitigated ranking;
2. explicit identity-irrelevance instruction when identity is task-irrelevant;
3. counterfactually-fair prompting inspired by prior work;
4. proposed preference-aligned counterfactual reranking (PAIR).

Primary endpoint: a utility–fairness Pareto analysis, not a claim that fairness should be maximized regardless of recommendation quality.

---

## 3. Benchmark design

### 3.1 Two complementary tracks

#### Track A — Personality-grounded recommendation

1. **GroupLens Personality 2018** — movie recommendation with measured Big Five variables.
2. **Music Master / Beyond the Big Five** — 279 participants, 745 songs, BFI/BFI-2 personality information and ratings.
3. **REASONER** — short-video recommendation containing interaction data plus a Big Five file based on CBF-PI-15 responses.

Purpose: answer whether *measured* psychographic context has incremental value beyond observed preferences.

#### Track B — Counterfactual generalization

4. **MovieLens-1M** — movie interaction history; demographics can support controlled demographic tests subject to the exact original schema/license.
5. **Last.fm-1K / HetRec Last.fm** — music listening histories; Last.fm-1K also exposes profile fields such as gender/age/country. Freeze one variant after license/schema verification and do not combine variants silently.
6. **MIND** — news recommendation with real click histories and logged candidate impressions; identity variables are not supplied, so identity manipulations are explicitly synthetic counterfactual stress tests rather than observed-user demographic analyses.

Purpose: stress-test whether the framework generalizes beyond the three personality datasets.

### 3.2 Dataset governance

Every adapter must emit a dataset card containing:
- source/version/download date;
- license;
- user/item/interactions counts after filtering;
- which fields are observed, derived, or counterfactual;
- split protocol and hash;
- excluded records and reasons;
- missing demographic/personality field rates;
- leakage checks.

Do **not** infer protected characteristics from names or free text.

### 3.3 User sampling

Primary matrix proposal (subject to pilot power/cost calculation):
- 100 evaluable users per dataset;
- fixed history length where possible (e.g. up to 20 prior positives);
- 50-item auditable candidate set containing held-out positives plus sampled negatives;
- K = 10 primary;
- K ∈ {5, 10, 20} sensitivity analysis.

Use identical user/candidate instances for every model condition. Store instance IDs and hashes before model calls.

---

## 4. Six-family LLM panel

Target one text-capable snapshot from each independent family:

| Family | Candidate model for pilot | Freeze rule |
|---|---|---|
| OpenAI | GPT-5.6 Terra | record exact API model/version returned where available |
| Anthropic | Claude Sonnet 5 | use active non-deprecated API ID |
| Google | Gemini 3.8 Flash | stable `gemini-3.8-flash` |
| DeepSeek | DeepSeek V4 Flash | API `deepseek-v4-flash`; record backend version |
| Alibaba | Qwen 3.8 Max | pin documented API model ID |
| Meta | Llama 4 Maverick | pin exact checkpoint/provider |

Selection principle: family diversity and practical reproducibility, not leaderboard cherry-picking. A pilot can replace a model if access/cost/schema support is unsuitable, but replacements must happen **before** the main run.

Use provider-native structured JSON output if supported. Do not use web search, retrieval, or personalization memory during ranking generation.

---

## 5. Controlled prompting

### 5.1 Candidate-constrained ranking

Open-ended “recommend 25 artists/movies” generation confounds hallucination, popularity knowledge, item availability, and preference quality. The new benchmark gives the model:

- a user preference history;
- an auditable candidate set with item IDs + minimal textual metadata;
- optional demographic/personality context according to condition;
- an instruction to return exactly K unique candidate IDs as JSON.

This makes relevance measurable against held-out interaction labels.

### 5.2 Conditions

Use a field-based template so wording/order remain constant:

- `C0`: preference history only; identity/personality = `unspecified`.
- `C1`: observed demographic identity (dataset permitting).
- `C2`: paired counterfactual demographic identity.
- `C3`: true measured OCEAN profile.
- `C4`: shuffled OCEAN profile sampled from a different user.
- `C5`: one-trait OCEAN counterfactual; remaining traits fixed.
- `C6`: demographic + true personality intersectional condition on datasets that support both.

For Track B datasets without measured personality, never label synthetic personality as observed. Synthetic manipulations may be used only as clearly labeled stress tests.

### 5.3 Personality representation

Represent personality as standardized numeric OCEAN values, e.g.:

```json
{"openness":0.78,"conscientiousness":0.43,"extraversion":0.21,"agreeableness":0.62,"neuroticism":0.55}
```

Do not convert these to stereotype-heavy prose such as “you are a shy engineer.” For reader-facing interpretability, trait bins (low/mid/high) may be reported only after pre-specified thresholding.

---

## 6. Metrics

### 6.1 Utility / retrieval effectiveness

Primary:
- nDCG@10
- Recall@10

Secondary:
- MRR@10 or HitRate@10 (dataset dependent)

### 6.2 Behavioral-shift diagnostics

- Rank-Biased Overlap (RBO)
- Jaccard@K
- optional original SERP/PRAG metrics for continuity with prior FairEval

**Important:** these metrics quantify ranking change; they do not independently establish unfairness.

### 6.3 Fairness / counterfactual-effect metrics

#### Counterfactual Utility Gap (CUG)
For paired conditions `a` and `a'` on the same user/candidates:

`CUG_u(a,a') = U(R_u^a) - U(R_u^{a'})`.

Report signed mean, absolute paired gap, 95% CI, and distribution. A group-level value without paired uncertainty is insufficient.

#### Group Utility Disparity (GUD)
Difference/range of mean held-out utility across pre-specified demographic or personality strata, with uncertainty intervals.

#### Counterfactual Exposure Gap (CEG)
When reliable item/provider/category metadata exists, compare top-K discounted exposure across counterfactual identity conditions.

#### Invalid Output Disparity (IOD)
Difference in persistent invalid-output rate across conditions/groups. Parsing failure is a behavioral outcome and must not be silently deleted.

### 6.4 Personality-specific value

Define a transparent incremental statistic rather than immediately inventing another opaque “fairness score”:

`PVA = [U(true personality) - U(neutral)] - [U(shuffled personality) - U(neutral)]`

which simplifies to `U(true) - U(shuffled)` under matched instances. Report this paired effect with confidence intervals.

### 6.5 What happens to original PAFS

Do **not** use original PAFS as the primary fairness claim. Keep it only as a legacy/sensitivity baseline to show why pure stability can disagree with preference-conditioned evaluation. If a PAFS-v2 is later introduced, it must be motivated by explicit desirable properties and compared with simpler alternatives first.

---

## 7. Stochasticity and reproducibility

### Main run
- 3 independent generations per prompt-condition instance.
- Low/common randomness where provider APIs permit comparable control.
- Record temperature, top-p, max output tokens, reasoning/thinking setting, seed support, API model ID, timestamp, request hash, prompt hash, and code commit.

### Stochasticity audit
- stratified 10% of instances;
- 10 repeated generations per condition;
- analyze within-instance variance and whether conclusions change across repeats.

If a provider does not honor deterministic seeds, report this as such rather than pretending cross-provider seed equivalence.

### Invalid outputs

1. JSON schema validation;
2. candidate-ID membership validation;
3. uniqueness and K validation;
4. one deterministic repair attempt that changes *format instructions only*, not recommendation content;
5. persistent failure retained in logs and included in IOD.

Primary results never silently drop failed generations.

---

## 8. Statistics

Pre-specify before full execution:

- paired bootstrap 95% CIs at the user level;
- paired permutation or Wilcoxon signed-rank tests where appropriate;
- Holm correction within each RQ family;
- standardized paired effect sizes;
- mixed-effects analysis for heterogeneity, e.g. `utility ~ condition * model + dataset + (1|user) + (1|template)` where assumptions/data size permit;
- variance decomposition for generation/template/model effects;
- utility–fairness Pareto curves for RQ4.

Report uncertainty and effect sizes, not significance stars alone.

---

## 9. Mitigation: PAIR

Working method name: **PAIR — Preference-Aligned Identity Re-ranking**.

Goal: preserve item relevance/preferences while penalizing ranking changes that are uniquely caused by task-irrelevant identity substitutions.

A simple implementation can operate on an ensemble of neutral and counterfactual model rankings:

`score(i) = relevance_consensus(i) - lambda * counterfactual_instability(i)`

where relevance consensus rewards items consistently ranked under preference-preserving contexts and the instability term penalizes identity-specific exposure swings. `lambda` is selected on validation data; report a Pareto frontier rather than one cherry-picked setting.

Important novelty boundary: prior work already studies counterfactually fair prompting and CFaiRLLM already links fairness evaluation to true preference alignment. PAIR must therefore be evaluated as one component of a larger contribution centered on **measured personality + negative controls + broad multi-model/domain reliability + preference-conditioned diagnosis**, not sold as novelty merely because it combines utility and fairness.

---

## 10. Planned figures and tables

### Main figures
1. **Conceptual quadrant:** ranking change × preference consequence: stable, beneficial personalization, unjustified sensitivity, harmful disparity.
2. **FairEval pipeline:** dataset → held-out user instance → controlled condition generator → six LLM adapters → validator → utility/fairness/reliability analysis.
3. **RQ2 true-vs-shuffled personality effect:** paired effect/forest plot by dataset/model/OCEAN trait.
4. **RQ3 reliability:** variance components / effect-size forest plot across models and domains.
5. **RQ4 Pareto frontier:** utility versus counterfactual gap for unmitigated / prompt / PAIR variants.

### Main tables
1. Six-dataset characteristics, observed variables, domains, and track assignment.
2. Six-model reproducibility manifest (exact IDs and settings).
3. RQ1 primary paired utility/fairness effects with CIs.
4. RQ2/RQ3 condensed effect summary.
5. RQ4 mitigation utility/fairness trade-off.

No major result should exist only in an appendix; ECIR appendices consume the 12-page limit anyway.

---

## 11. 12-page ECIR story

Suggested allocation (references excluded):

1. Introduction + contributions — 1.2 pages
2. Related work — 1.1
3. Problem formulation / personalization-vs-discrimination framework — 1.3
4. FairEval benchmark and controlled design — 2.1
5. Experimental setup — 1.6
6. Results RQ1–RQ4 — 3.5
7. Discussion, limitations, ethics / positive social outcome — 0.8
8. Conclusion — 0.4

This time RQs appear in the introduction **before** methodology.

---

## 12. Contribution claims we may make only after results exist

Safe design-level claims now:

1. A preference-conditioned framework that explicitly separates ranking sensitivity from fairness consequences.
2. A benchmark design grounding personality in validated Big Five measurements with shuffled and counterfactual controls.
3. A six-dataset/six-family reliability protocol with repeated generations and auditable candidate-constrained ranking.
4. A mitigation evaluation that measures fairness jointly with recommendation utility.

Do **not** write “FairEval improves fairness,” “personality improves recommendation,” or “model X is fairest” until generated results support those statements.

---

## 13. Relevant verified anchors for the literature review

- Zhang et al., *Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation*, RecSys 2023, DOI: `10.1145/3604915.3608860`.
- Hua et al., *UP5: Unbiased Foundation Model for Fairness-aware Recommendation*, EACL 2024, DOI: `10.18653/v1/2024.eacl-long.114`.
- Deldjoo & Di Noia, *CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System*, ACM TIST 16(6), 2025, DOI: `10.1145/3725853`.
- Kantharuban et al., *Stereotype or Personalization? User Identity Biases Chatbot Recommendations*, Findings of ACL 2025, DOI: `10.18653/v1/2025.findings-acl.1254`.
- Neplenbroek et al., *Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization*, EMNLP 2025, DOI: `10.18653/v1/2025.emnlp-main.1029`.
- Kleć et al., *Beyond the Big Five Personality Traits for Music Recommendation Systems*, EURASIP Journal on Audio, Speech, and Music Processing, 2023 dataset/paper; verify final bibliographic fields before manuscript freeze.

All bibliography entries must be independently verified before final submission; never cite an unverified reference generated from memory.
