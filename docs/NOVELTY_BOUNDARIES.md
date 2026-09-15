# ECIR 2027 Novelty Boundaries

The ECIR study must be positioned against both external literature and the authors' own prior publications. This document prevents accidental re-claiming of older contributions.

## 1. Prior author work that constrains novelty

### FairEval (2025 preprint / 2026 published chapter)
Already contains:
- FairEval framework name;
- demographic + personality-aware fairness framing;
- PAFS@25;
- ChatGPT/Gemini evaluation;
- movie/music tasks;
- structured identity/personality prompt variants;
- prompt robustness analyses.

**ECIR implication:** none of these can be the central novelty.

### PerFairX (ICCV Workshops 2025)
Already contains:
- OCEAN/personality-aware LLM recommendation;
- an explicit fairness--personality / personalization trade-off framing;
- ChatGPT and DeepSeek;
- MovieLens 10M and Last.fm 360K;
- comparison of neutral and personality-sensitive prompts.

**ECIR implication:** “Big Five + fairness/utility trade-off” alone is not new.

### Uncertainty and Fairness Awareness in LLM-Based Recommendation Systems (2026 public work)
This line couples fairness with uncertainty/trustworthiness. Unless the ECIR study executes a genuinely new uncertainty experiment, uncertainty should not be added opportunistically to the ECIR contribution list.

---

## 2. External literature that constrains novelty

### FaiRLLM (RecSys 2023)
Already establishes sensitive-attribute perturbation for RecLLM fairness.

### UP5 (EACL 2024)
Already establishes counterfactually fair prompting as a mitigation direction.

### CFaiRLLM (ACM TIST 2025)
Already argues that recommendation-list differences can reflect legitimate personalization and incorporates true-preference alignment/intersectional fairness.

### Stereotype or Personalization? (Findings ACL 2025)
Already establishes that user identity can affect recommendations in ways mixing personalization and stereotyping.

### Reading Between the Prompts (EMNLP 2025)
Already establishes stereotype-driven implicit personalization from subtle demographic cues and demonstrates intervention-based mitigation.

### 2026 LLM4Rec fairness survey
The emerging literature now explicitly organizes fairness by bias mechanisms, fairness targets, evaluation resources, utility--fairness trade-offs, mitigation, robustness, and controllability. A narrow prompt-sensitivity benchmark is therefore no longer competitive as a full-paper contribution.

---

## 3. What the ECIR paper is actually about

### Core novelty statement

> **A preference-conditioned, counterfactual, psychometrically grounded reliability benchmark that diagnoses whether contextual ranking changes are justified by user preference evidence, rather than treating either list invariance or personalization gains as fairness by themselves.**

This requires all of the following to be true in the executed paper:

1. **Psychometric ground truth** — measured Big Five/OCEAN/BFI-2 data, not OCEAN inferred from genres or synthetic persona adjectives.
2. **Matched negative control** — true profile versus shuffled profile from another user.
3. **Trait intervention** — one OCEAN dimension changed at a time with other traits fixed.
4. **Paired demographic intervention** — same user/history/candidate set, counterfactual identity field changed only.
5. **Real held-out utility** — nDCG/Recall from observed interactions or impression labels.
6. **Candidate-constrained generation** — no open-ended artist/director list where hallucination/popularity confounds relevance.
7. **Two-axis diagnosis** — behavioral shift and utility/exposure consequence reported jointly.
8. **Repeated sampling** — not single-pass APIs.
9. **Reliability decomposition** — quantify model, dataset/domain, template, cutoff, and generation variance.
10. **Failure as outcome** — invalid outputs reported rather than dropped.
11. **Broad model/domain panel** — six independent LLM families across six datasets.
12. **Mitigation that preserves legitimate personalization** — utility/fairness Pareto analysis rather than fairness-only optimization.

If several of these are removed, the paper risks collapsing back into previously published FairEval/PerFairX territory.

---

## 4. Claim-by-claim guard

| Candidate ECIR claim | Allowed? | Reason |
|---|---:|---|
| “We introduce FairEval.” | No | Already public/published. |
| “We are first to study personality-aware fairness in RecLLM.” | No | FairEval and PerFairX already do this. |
| “We show a fairness--personality trade-off.” | No as novelty | PerFairX already frames this problem. |
| “We distinguish personalization from fairness.” | Not alone | CFaiRLLM and ACL 2025 already motivate this distinction. |
| “We use true preferences when judging fairness.” | Not alone | CFaiRLLM already moves in this direction. |
| “We ground LLM personality evaluation in measured Big Five profiles and matched shuffled controls.” | Potentially | Stronger and more specific; must verify no closest prior work duplicates the full design. |
| “We provide a paired preference-conditioned counterfactual audit with six-family/six-dataset reliability decomposition.” | Potentially strong | Combination changes causal control and reliability scope. |
| “We quantify invalid-output disparity and generation-level variance as fairness reliability dimensions.” | Potentially strong | Must be supported by executed analysis. |
| “We mitigate identity effects while preserving personality-derived utility using a Pareto evaluation.” | Potentially strong | Must outperform or complement existing prompting baselines without overclaiming novelty of counterfactual prompting. |

---

## 5. Recommended paper identity

For anonymous review, avoid making the public FairEval/PerFairX lineage the title hook. Recommended title:

> **Separating Personalization from Discrimination in Personality-Aware LLM Recommendations**

Even more novelty-forward alternative:

> **Preference-Conditioned Counterfactual Auditing for LLM Recommendations**

The second title is scientifically cleaner if the final experiments emphasize causal controls and reliability more than personality branding.
