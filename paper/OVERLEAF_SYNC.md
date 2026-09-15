# Overleaf Sync Checklist

The canonical ECIR 2027 LaTeX source is this `paper/` directory on branch `ecir-2027-redesign`. Keep the Overleaf project synchronized with these files as one unit:

- `main.tex`
- `benchmark_model_table.tex`
- `rq_design_table.tex`
- `results_contract_table.tex`
- `related_work_table.tex`
- `references.bib`
- `references_extra.bib`
- `figures/faireval_framework.pdf`
- `figures/faireval_conditions.pdf`
- `figures/faireval_evaluation_pipeline.pdf`

Current frozen scope: six real-world datasets + separately reported FairSynth-360; six hosted/API families + two local open-weight models; contextual PAIR with one global validation-frozen operating point; no test-set hyperparameter selection.

Before an Overleaf upload, run from the repository root:

```bash
pytest -q
python scripts/check_config_consistency.py
python scripts/build_paper_figures.py
python scripts/check_paper_assets.py
python scripts/check_paper_source.py
```

Do not manually enter empirical numbers into the LaTeX. Result tables and result PDFs must be produced from frozen analysis artifacts. See `docs/OVERLEAF_SYNC.md` for the full synchronization contract.
