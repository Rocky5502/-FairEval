# ECIR 2027 submission compliance guardrails

This file records conference constraints that must be checked before the anonymous submission is exported. It is intentionally separate from the scientific design so formatting/anonymity mistakes cannot be hidden inside the manuscript workflow.

## Track

Current target: **ECIR 2027 Full Paper Track**.

Why the main full-paper track is defensible: the work contributes to recommender systems, LLM-based recommendation, personalization, evaluation methodology, ranking, and fairness. The IR-for-Good track remains a plausible alternative, but the project is currently being engineered as a full research paper rather than relying on a social-impact framing alone.

## Deadlines

- Abstract registration: **21 September 2026, 23:59 GMT**.
- Full paper: **5 October 2026, 23:59 GMT**.
- Notification: **7 December 2026**.

The registered title and abstract must remain representative of the final paper because they are used for reviewer bidding.

## Length and template

- Springer LNCS proceedings format.
- Maximum **12 pages**, plus unlimited reference pages.
- Appendices count inside the 12-page limit and must appear before references if used.
- Therefore the submission must be self-contained: core RQs, prompt design, metrics, dataset/model table, statistical protocol, primary results, and major limitations must remain in the main 12 pages.
- Do not shrink fonts, margins, captions, or spacing to bypass the limit.

## Double blindness

- Anonymous authors and affiliations in the review PDF.
- Remove acknowledgements, grant identifiers that reveal identity, repository owner names, institutional URLs, author-specific ORCIDs, and self-identifying artifact links.
- Repository/artifact release for review must use an anonymous mechanism if a link is included.
- Self-citations to prior published work should be written in the third person and only when scientifically necessary.
- Do **not** cite the existing FairEval arXiv technical report in the anonymous ECIR manuscript.

## Existing arXiv report and title risk

The prior FairEval version is publicly available on arXiv. ECIR permits previously available technical reports, but explicitly warns authors to protect double blindness. Reusing the exact public title increases deanonymization risk even when reviewers are instructed not to break anonymity.

Project decision as of 2026-09-15:

- Preserve **FairEval** as the research identity requested by the lead author.
- Internal/camera-ready target title: **FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness**.
- Before abstract registration, make one explicit submission decision:
  1. use the exact FairEval title and knowingly accept the anonymity risk; or
  2. use a materially representative anonymous review title and restore the FairEval title after acceptance if chairs permit.
- This decision must not change the scientific scope or be used to misrepresent overlap with the technical report.

## Prior-publication / substantial-difference guard

The ECIR submission must be substantially different from the old arXiv/book versions. Minimum differences required before submission:

1. real preference histories and candidate-constrained ranking instead of open-ended synthetic recommendation prompts;
2. psychometrically measured Big Five personality rather than informal personality adjectives;
3. explicit separation of demographics and psychographics;
4. paired counterfactual utility/exposure gaps rather than treating list change as unfairness;
5. shuffled-personality and one-trait negative controls;
6. six datasets and six model families with pinned/versioned manifests;
7. repeated generations plus prompt/candidate-order reliability analysis;
8. explicit invalid-output accounting;
9. utility--fairness mitigation evaluation, including PAIR;
10. new conclusions based only on the redesigned experiments.

If the final paper falls back to the old two-model/two-domain sensitivity story, it should **not** be submitted as the redesigned ECIR paper.

## Result-integrity guard

- No invented result values.
- No copying pilot numbers into the confirmatory table without a clearly labeled pilot status.
- No selecting prompts, K, models, datasets, or mitigation hyperparameters after seeing test results unless disclosed as exploratory.
- Main LaTeX tables/figures should be generated from frozen result artifacts, with hashes recorded.
- Any failed or malformed generation remains observable system behavior and must not be silently excluded.

## Final pre-submission checklist

- [ ] abstract registration completed by deadline
- [ ] exact track confirmed
- [ ] title/anonymity decision explicitly frozen
- [ ] Springer LNCS template is official/current
- [ ] review PDF is anonymous
- [ ] <= 12 pages before references, including appendix
- [ ] no citation to the FairEval arXiv report
- [ ] model IDs and provider versions frozen
- [ ] dataset licenses/source versions documented
- [ ] primary prompt/config hashes frozen
- [ ] pilot and confirmatory data separated correctly
- [ ] all primary result tables generated from result files
- [ ] RQ1--RQ4 each answered by evidence, not intention text
- [ ] limitations and ethical scope included
- [ ] repository/anonymized artifact does not reveal authors
