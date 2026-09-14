# ECIR 2027 manuscript

This folder contains the result-honest ECIR manuscript scaffold.

## Template

ECIR 2027 requires the Springer proceedings template. Download the official **LaTeX2e Proceedings Template** from Springer's LNCS author page / Overleaf and place the official template files (including `llncs.cls`) in this folder before compiling.

The repository intentionally does not vendor an unofficial or modified copy of the Springer class.

## Anonymous submission

ECIR Full Papers use double-blind review. `main.tex` therefore contains anonymous author metadata. Author names/affiliations are added only for camera-ready.

## Result integrity

All numerical cells marked `TBD` are placeholders, not findings. Before submission:

1. run the frozen experiment matrix;
2. generate aggregate CSV/JSON from immutable raw response manifests;
3. generate LaTeX result tables and plot PDFs from scripts;
4. replace `TBD` only through generated artifacts or verified values;
5. cross-check every abstract/conclusion number against the generated results.

Do not manually invent or estimate result values.

## Suggested build

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or import the folder into Overleaf with the official LNCS template files.
