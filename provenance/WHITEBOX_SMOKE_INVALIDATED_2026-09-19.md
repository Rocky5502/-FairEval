# White-box engineering smoke invalidation — 2026-09-19

## Scope

The early local GPU launcher `scripts/run_whitebox_campaign.py` was used only as an
engineering smoke path while validating Windows/CUDA/model-download behavior.

That launcher did **not** execute the immutable FairEval run plan, did not instantiate
the six registered FairSynth conditions, and did not persist canonical
`planned_cell_id` provenance. Its local outputs therefore cannot support ECIR
fairness, personality, robustness, or white-box empirical claims.

## Observed engineering progress

The smoke path demonstrated that:

- Qwen/Qwen2.5-7B-Instruct could be downloaded and executed on the workstation;
- mistralai/Mistral-7B-Instruct-v0.3 could be downloaded and executed after the
  missing protobuf dependency was installed;
- microsoft/Phi-3.5-mini-instruct downloaded successfully but exposed a
  `DynamicCache.seen_tokens` compatibility failure with the then-installed
  Transformers version.

These observations are infrastructure/debugging evidence only. The Mistral smoke
model is not part of the canonical ECIR local panel.

## Canonical replacement

The canonical local study uses exactly:

- `qwen25_local`: Qwen/Qwen2.5-7B-Instruct;
- `phi35_local`: microsoft/Phi-3.5-mini-instruct.

The immutable FairSynth geometry remains:

`360 users × 6 registered conditions × 3 seeded repetitions × 2 models = 12,960 cells`.

The canonical 16 GB workstation path freezes bitsandbytes NF4 4-bit loading with
bfloat16 compute, one model at a time. On the pinned Transformers 4.44.2 stack,
both canonical models keep KV caching enabled; Phi-3.5 uses eager attention. These runtime policies are included in the
scientific specification and require a fresh pre-execution seal before canonical
GPU execution.

## Reporting rule

Files produced by the retired generic-prompt smoke launcher must not be passed to
`scripts/finalize_whitebox.py`, copied into paper tables, plotted as FairEval
results, or mixed with canonical run logs. Canonical paper artifacts must originate
only from audited logs created by `scripts/run_whitebox_family.py` or
`scripts/run_whitebox_full.py` against a matching seal and immutable plan.
