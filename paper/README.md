# ECIR 2027 manuscript

This folder contains the result-honest, double-blind ECIR 2027 manuscript source.

## Template

ECIR 2027 uses the Springer proceedings/LNCS template. Obtain the official LaTeX2e Proceedings Template from Springer/Overleaf and use the official `llncs.cls`/`splncs04` support files when compiling. The repository intentionally does not vendor an unofficial or modified Springer class.

## Canonical Overleaf workflow

The repository `paper/` directory on branch `ecir-2027-redesign` is the scientific source of truth. Do not maintain independent method definitions only in Overleaf.

Run the manuscript/config checks and create an anonymous upload archive with:

```bash
python scripts/check_config_consistency.py
python scripts/build_paper_figures.py
python scripts/check_paper_assets.py
python scripts/check_paper_source.py
python scripts/build_overleaf_bundle.py \
  --output dist/FairEval_ECIR2027_Overleaf.zip
```

CI runs these guards and publishes the generated ZIP as the `FairEval-ECIR2027-Overleaf` workflow artifact. See `OVERLEAF_SYNC.md` and `../docs/OVERLEAF_SYNC.md` for the synchronization contract.

The bundle intentionally contains the anonymous LaTeX sources, tables, bibliographies and methodology figures—not API credentials, raw datasets, experiment logs or author-identifying repository metadata. Generated RQ result figures are included only when their frozen analysis artifacts already exist.

## Current manuscript scope

- six real-world datasets for real-world fairness/personality claims;
- FairSynth-360 as a separately analyzed controlled synthetic sanity benchmark;
- six hosted/API model families plus two local open-weight configurations;
- Qwen2.5-7B and Phi-3.5-mini local diagnostics explicitly labeled exploratory/uncalibrated;
- contextual PAIR with one global operating point selected on a deterministic 20% validation split, requiring at least 95% of unmitigated validation nDCG before minimizing absolute CUG;
- no test-set or per-model/per-dataset PAIR tuning.

## Anonymous submission

ECIR Full Papers use double-blind review. `main.tex` therefore contains anonymous author metadata. Author names/affiliations are added only when the review policy permits de-anonymization.

## Result integrity

All numerical cells marked `TBD` are placeholders, not findings. Before submission:

1. run the frozen experiment matrix;
2. generate aggregate JSON/JSONL from immutable raw response manifests;
3. generate LaTeX result tables and vector-PDF result figures from scripts;
4. populate result cells only through generated artifacts;
5. cross-check every abstract/conclusion number against those artifacts.

Do not manually invent or estimate result values.

## Suggested build

With the official Springer class/support files available:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or upload `dist/FairEval_ECIR2027_Overleaf.zip` to the Overleaf project and add the official Springer template support files as required.
