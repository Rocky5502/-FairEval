# ECIR 2027 page-budget and float audit

Status date: 2026-09-24

This note records a real Springer LNCS compile audit of the anonymous final-results
manuscript. It is a layout/reproducibility record, not an empirical result.

## Current audited compile

The exact CI-generated Overleaf bundle at the compact final-results state was
compiled with:

```bash
pdflatex main.tex
bibtex8 main
pdflatex main.tex
pdflatex main.tex
```

Audit result:

- **10 PDF pages total**, including references;
- References begin on page 8, leaving the scientific body comfortably below a
  12-content-page ceiling;
- the framework figure, primary FairSynth paired-effect table, V7 secondary
  figure, repetition-stability table, and hosted six-family radar profile all
  occur before the bibliography;
- no undefined citations or references;
- hyperlink boxes are hidden for the anonymous submission;
- the final render contains no clipping, overlap, broken glyphs, or unreadable
  result-table text.

## Compact body policy

The main paper deliberately keeps only evidence that directly answers the
completed RQs:

1. FairEval consequence-aware framework;
2. primary paired FairSynth utility table;
3. frozen-result condition/effect-distribution figure;
4. two-repetition stability table;
5. hosted six-family operational coverage radar.

The following remain packaged in the repository/Overleaf bundle for auditability
but are not typeset in the page-limited body:

- large literature comparison matrix;
- benchmark/model configuration table;
- condition schematic;
- artifact-to-claim pipeline schematic;
- auxiliary hosted/white-box/execution-quality tables;
- deferred real-world and mitigation result contracts.

CI rejects accidental reintroduction of those auxiliary assets into
`paper/main.tex`.

## Scientific layout rules

Do not recover space by shrinking global fonts or deleting uncertainty
information. Prefer:

- concise prose over duplicate tables;
- one primary effect table plus figures that add distributional/operational
  information;
- short captions with explicit interpretation guards;
- provenance detail in the anonymized artifact rather than the narrative;
- readable single-column LNCS floats using `\linewidth`.

## Final acceptance criteria

Before submission:

1. build only from a green `ecir-2027-final-results` head;
2. regenerate the hosted and frozen V7 secondary artifacts;
3. run `scripts/check_submission_closeout.py`;
4. compile the exact CI-generated Overleaf ZIP under LNCS/BibTeX;
5. confirm total/content page count remains inside the venue limit;
6. render every PDF page at >=160 DPI and inspect for overlap/clipping;
7. confirm no undefined citations/references and no material overfull boxes;
8. upload only the freshly hashed final bundle.
