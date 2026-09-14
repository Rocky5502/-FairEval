# Prior-Publication / Double-Blind Guard for ECIR 2027

## Why this file exists

The ECIR manuscript must not become a lightly expanded version of prior FairEval outputs.

Known prior public versions include:

1. **FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness** — arXiv:2504.07801 / public preprint.
2. **Fairness Evaluation in LLM-Based Educational Recommender Systems: A Framework for Equitable Learning** — IntechOpen book chapter, published 17 February 2026, DOI `10.5772/intechopen.1012999`.

The published chapter describes FairEval with eight demographic attribute categories, personality profiles, PAFS@25, ChatGPT/Gemini evaluation, movie/music tasks, and >1,000 structured personas. Those elements therefore cannot be presented as new contributions at ECIR.

ECIR 2027's full-paper policy states that submitted papers must be substantially different from papers previously published, accepted, or under review. ECIR permits arXiv technical reports but asks authors to preserve double blindness; published work is not covered by the technical-report exception.

## Internal project name vs. submission title

The repository can remain **FairEval** for continuity. For anonymous ECIR review, the safest working title is:

> **Separating Personalization from Discrimination in LLM Recommendations**

Alternative with a stronger personality signal:

> **Preference-Conditioned Fairness for Personality-Aware LLM Recommendation**

After acceptance, the camera-ready title can be reconsidered with the chairs if retaining the FairEval lineage is scientifically and ethically appropriate.

## What the ECIR paper must NOT claim as new

- introducing the name FairEval;
- adding personality in the abstract sense;
- the original PAFS@25 formulation;
- the original eight sensitive-attribute prompt benchmark;
- GPT/Gemini movie/music comparison;
- typo/French prompt sensitivity from the older evaluation;
- the existence of an LLM recommendation fairness audit itself.

## What makes the proposed ECIR study substantially different

The ECIR study changes the **scientific object, data, causal controls, metrics, scale, and intervention question**:

1. **New problem definition:** ranking sensitivity is separated from fairness consequence by conditioning on true preference utility.
2. **Real user grounding:** recommendation tasks are built from interaction histories and held-out relevance labels, not hand-curated open-ended artist/director prompts.
3. **Validated personality:** three personality-grounded datasets use measured Big Five/OCEAN/BFI-2 style instruments.
4. **Negative controls:** true personality is compared against shuffled personality and one-trait counterfactual edits.
5. **Counterfactual controls:** identity interventions hold user history, candidate set, task, and prompt syntax fixed.
6. **New endpoints:** nDCG/Recall + counterfactual utility/exposure gaps + invalid-output disparity; old PAFS is only a legacy diagnostic.
7. **Scale/generalization:** six datasets and six independent LLM families.
8. **Reliability:** repeated generations, prompt-template variance, K sensitivity, provider/model-version manifests, and failure-rate analysis.
9. **Mitigation:** utility-preserving prompting/reranking (PAIR) is evaluated rather than audit-only conclusions.
10. **New statistical design:** paired inference, confidence intervals, effect sizes, multiplicity correction, and heterogeneity analysis.

## Manuscript overlap rules

- Do not copy paragraphs, tables, figures, equations, or result prose from the book chapter/preprint.
- Rebuild the introduction from the new problem definition.
- Cite the published predecessor in the final scholarly record as required; write any self-citation neutrally and follow ECIR's double-blind guidance/chair instructions.
- Do not cite the arXiv technical report in the anonymous ECIR submission if the venue instructions advise against it.
- Do not reuse old numerical results as ECIR evidence except in a clearly identified historical comparison, if allowed and scientifically needed.
- Maintain a final `prior_work_diff.md` table for the submission package showing old vs. new research questions, data, models, metrics, and contribution claims.

## Go/no-go criterion

Do **not** submit the ECIR paper if the final experiment collapses back to the old two-model/two-domain prompt-sensitivity audit. The paper is submission-worthy only if the preference-grounded, measured-personality, counterfactual, repeated-generation, multi-family design is actually executed and reported.
