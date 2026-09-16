# ECIR 2027 page-budget and float audit

Status date: 2026-09-16

This file records a real Springer LNCS compile audit of the anonymous development manuscript. It is a layout-engineering note, not an empirical result.

## Current compile state

The green Overleaf bundle compiles through:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The audited development PDF is currently 17 pages total. The Conclusion and References begin around page 13, so the scientific body remains slightly above a strict 12-page target once all floats are placed in source order.

## Important float finding

The current source order is correct, but LaTeX can defer three large tables to a final float page after the bibliography:

- the RQ design/estimand table;
- Main Table M1 (RQ1--RQ2);
- Main Table M2 (RQ3--RQ4).

All three fit cleanly on one float page in the current TBD development state. The table designs themselves are therefore compact; the remaining problem is placement and overall page pressure, not table width.

Do not treat a final float page after References as submission-ready. Before final submission, recover enough body space and ensure the result anchors occur before the bibliography. A blind `\clearpage` only exposes the true page pressure; it does not solve it.

## Result-table policy

Always keep as main-paper anchors:

1. `paper/result_tables/rq12_main_table.tex`
2. `paper/result_tables/rq34_main_table.tex`

Evidence-ready compact tables, promoted selectively:

- execution coverage / invalid-output rate;
- RQ2 C5 one-trait robustness;
- FairSynth-360 sanity checks;
- local white-box diagnostics.

## Figure budget

The repository can generate four RQ result figures, but the 12-page paper does not need all four merely for symmetry. Prefer figures that add information not already compactly represented in the two main tables. The likely high-value visual anchors are the RQ1 consequence geometry and RQ4 utility--fairness frontier; the final decision must follow the actual evidence.

Do not delete figure-generation code simply because a figure is omitted from the main 12-page version.

## Compression priorities

Prefer scientific/layout compression over unreadably small text:

- shorten repetitive related-work prose because the audited comparison table carries much of the design-coverage detail;
- consolidate repeated dataset/model caveats into table notes;
- keep only result figures that provide non-tabular evidence;
- shorten captions without removing interpretation guards;
- remove development-only explanatory sentences after provenance is frozen;
- fix long unbreakable phrases rather than shrinking global fonts.

## Final acceptance criteria

Before submission:

1. compile the exact CI-generated Overleaf bundle using LNCS/BibTeX;
2. verify the main text, including tables and figures, satisfies the ECIR page limit;
3. verify both main result tables occur before References;
4. render the final PDF at >=160 DPI and inspect every page for clipping, overlap, broken glyphs, or unreadable table text;
5. ensure there are no undefined citations/references and no material overfull boxes;
6. regenerate the upload ZIP only from the final green repository head.
