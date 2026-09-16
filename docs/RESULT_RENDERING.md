# FairEval result rendering

Paper-facing empirical numbers must be generated from frozen analysis artifacts. Do not hand-edit result cells.

## Canonical command

Use the LNCS-safe wrapper, **not** the lower-level renderer directly:

```bash
python scripts/render_result_tables_lncs.py \
  --inference-jsonl results/analysis/core/inference.jsonl \
  --trait-inference-jsonl results/analysis/rq2_traits/inference.jsonl \
  --rq3-summary-jsonl results/analysis/rq3/variation_summary.jsonl \
  --rq4-artifact-json results/analysis/rq4/rq4_pair_artifact.json \
  --rq4-prompting-summary-json results/analysis/rq4_prompting/prompting_summary.json \
  --fairsynth-inference-jsonl results/analysis/fairsynth/inference.jsonl \
  --whitebox-summary-jsonl results/analysis/local_whitebox/summary.jsonl \
  --user-condition-jsonl results/analysis/core/user_condition.jsonl \
  --output-dir paper/result_tables
```

The wrapper first runs `scripts/render_result_tables.py`, which validates all required artifact families and renders the two main + four compact tables. It then normalizes the two main tables to ordinary `table` floats required by the single-column Springer LNCS layout.

After rendering:

```bash
python scripts/check_result_table_contracts.py
python scripts/check_paper_source.py
python scripts/build_overleaf_bundle.py --output dist/FairEval_ECIR2027_Overleaf.zip
```

Compile and visually inspect the exact generated Overleaf bundle before submission. The four compact tables remain evidence-ready even when page budget permits only a subset in the final 12-page body.
