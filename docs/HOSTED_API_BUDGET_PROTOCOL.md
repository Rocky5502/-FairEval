# FairEval ECIR 2027 — Hosted API Budget Protocol

## Scope

This protocol applies to the six-family black-box track executed through the
Zhizengzeng OpenAI-compatible gateway. It does not change the local/open-weight
transparency track; that track is optional for the current hosted-only phase.

## Budget contract

- Planning/alert target: **200 RMB**
- Hard ceiling: **250 RMB**
- Pre-request reserve: **2 RMB**
- Source of record for spend: gateway balance movement from the experiment's
  initial balance, not a hand-computed token estimate.
- The hard ceiling is not relaxed after looking at results. A different ceiling
  requires a new ledger/experiment version.

Zhizengzeng exposes the account balance at
`POST /v1/dashboard/billing/credit_grants`. The runner records the initial
`available_amount`, checks the balance before and after every hosted cell, and
stops before launching another request when the remaining experiment budget is
less than or equal to the frozen reserve.

The reserve is deliberately conservative for FairEval's short,
candidate-constrained ranking calls. It bounds the risk that the final in-flight
request is charged after the last pre-request balance check.

## Hosted model freeze (2026-09-17)

The current gateway-specific panel is:

| Family | Frozen gateway model ID |
| --- | --- |
| OpenAI | `gpt-5.6-terra` |
| Anthropic | `claude-sonnet-5` |
| Google | `gemini-3.8-flash` |
| DeepSeek | `deepseek-v4.1-flash` |
| Alibaba Qwen | `qwen3.8-2.4t-a95b` |
| Meta | `llama-4-maverick` |

A live `GET /v1/models` preflight on the experiment key is still required before
paid execution. If any frozen ID is absent, execution stops. Do not substitute a
nearby model after examining results.

## Execution

The runner is dry-run by default:

```powershell
python scripts/run_hosted_budgeted.py `
  --plan-dir results/plans/hosted-core-v1 `
  --freeze-root data/frozen `
  --output-jsonl results/runs/hosted-core-v1.jsonl
```

After plan and model-list review, real execution requires `--execute`:

```powershell
python scripts/run_hosted_budgeted.py `
  --plan-dir results/plans/hosted-core-v1 `
  --freeze-root data/frozen `
  --output-jsonl results/runs/hosted-core-v1.jsonl `
  --ledger results/budget/hosted_zzz_v1.json `
  --target-rmb 200 `
  --hard-cap-rmb 250 `
  --request-reserve-rmb 2 `
  --execute
```

For the first paid run, add `--max-cells 12` and inspect the persisted rows,
invalid-output accounting, model resolution, and balance ledger before scaling.
Resume semantics are exact: already persisted `planned_cell_id` values are not
regenerated.

## Paper synchronization

Numerical paper content must come from audited artifacts. The intended chain is:

```text
hosted JSONL
  -> run-log audit
  -> RQ analysis artifacts
  -> render_result_tables.py / render_result_tables_lncs.py
  -> build_result_figures.py
  -> render_paper_results.py
  -> paper/main.tex / Overleaf bundle
```

Do not type a result value into the manuscript by hand. If a cell has not been
executed and audited, it remains explicitly pending rather than being inferred
or back-filled.
