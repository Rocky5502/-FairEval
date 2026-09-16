# Paper result tables

This directory contains the frozen paper-facing result-table contract.

## Always in the main Results section

- `rq12_main_table.tex` — primary RQ1/RQ2 cross-model, cross-dataset effects.
- `rq34_main_table.tex` — RQ3 reliability + the three-method RQ4 mitigation summary.

`paper/results_contract_table.tex` inputs these two files, so the manuscript keeps a stable entrypoint while the table bodies can be regenerated from frozen artifacts.

## Compact supporting tables

- `coverage_table.tex`
- `trait_ablation_table.tex`
- `fairsynth_table.tex`
- `whitebox_table.tex`

These are included in every Overleaf bundle but are not automatically inserted into the main 12-page paper. After real results exist, promote only the compact tables that materially support the story; keep the rest available for author/reviewer inspection and future extended versions. ECIR appendices count toward the page limit, so moving a table to an appendix does not solve page pressure.

## RQ4 evidence contract

Big Table 2 requires three evidence paths:

1. unmitigated audit outcomes from the frozen core run;
2. identity-irrelevance prompting from a separate executor-compatible plan built by `scripts/build_rq4_prompt_plan.py` and analyzed by `scripts/analyze_rq4_prompting.py`;
3. contextual PAIR from the validation-frozen `rq4_pair_artifact.json`.

The mitigation prompt is never inserted into the RQ1 audit plan. The derived prompting plan retains `faireval-run-plan-v1`, candidate/model/decoding/repetition settings, and records its source audit cell ID; only the named prompt mode changes.

## Source of truth

Before results:

```bash
python scripts/render_result_tables.py --contracts-only
```

After frozen results:

```bash
python scripts/render_result_tables.py \
  --inference-jsonl <core inference.jsonl> \
  --trait-inference-jsonl <trait_inference.jsonl> \
  --rq3-summary-jsonl <rq3_variation_summary.jsonl> \
  --rq4-artifact-json <rq4_pair_artifact.json> \
  --rq4-prompting-summary-json <prompting_summary.json> \
  --fairsynth-inference-jsonl <fairsynth inference.jsonl> \
  --whitebox-summary-jsonl <whitebox_summary.jsonl> \
  --user-condition-jsonl <core user_condition.jsonl>
```

Do not hand-edit empirical values in these files. If a final artifact is unavailable, the final renderer must fail rather than silently invent, estimate, reuse a stale number, or leave a method without an evidence source.
