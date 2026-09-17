# FairEval ECIR 2027 — Hosted API Budget Protocol

## Scope

This protocol applies to the six-family black-box track executed through the
Zhizengzeng OpenAI-compatible gateway. The local/open-weight transparency track
remains available for later replication but is **not required** for the current
hosted-only phase.

The first executable hosted target is FairSynth-360, because it is project-owned
and already reproducibly generated/frozen. Real-world RQ1/RQ2 execution remains
blocked until the exact third-party releases are locally acquired, hashed, and
frozen under their dataset contracts.

## Budget contract

- Normal hosted stop target: **200 RMB**
- Emergency hard ceiling: **250 RMB**
- Pre-cell reserve: **2 RMB**
- Safety buffer between normal target and ceiling: **50 RMB**
- Source of record for spend: gateway balance movement from the experiment's
  initial balance, not a hand-computed token estimate.
- The policy is outcome-independent: cells run in immutable plan order and the
  budget is never expanded after examining empirical results.
- A different target/ceiling requires a new ledger/experiment version.

Zhizengzeng exposes account balance at
`POST /v1/dashboard/billing/credit_grants`. The runner records the initial
`available_amount`, checks balance before and after every hosted cell, and stops
launching new cells when the remaining amount to the **200 RMB target** is less
than or equal to the frozen 2 RMB reserve. The separate 250 RMB ceiling is an
emergency boundary and should remain unused under ordinary execution.

A FairEval cell can contain one format-only repair request. This is why we keep a
large 50 RMB target-to-ceiling safety buffer rather than attempting to spend all
the way to 250 RMB.

## Hosted model freeze (2026-09-17)

The gateway-specific panel is:

| Family | Frozen gateway model ID |
| --- | --- |
| OpenAI | `gpt-5.6-terra` |
| Anthropic | `claude-sonnet-5` |
| Google | `gemini-3.8-flash` |
| DeepSeek | `deepseek-v4.1-flash` |
| Alibaba Qwen | `qwen3.8-2.4t-a95b` |
| Meta | `llama-4-maverick` |

A live `GET /v1/models` preflight on the experiment key is mandatory before paid
execution. If any exact frozen ID is absent, execution stops. Do not substitute a
nearby model after examining results.

Because the common gateway does not prove each vendor-native thinking/reasoning
control, the hosted adapter records native deliberation/sampling application as
**unverified** unless returned gateway metadata proves otherwise. Temperature and
top-p are sent through the OpenAI-compatible interface and logged as requested,
not automatically claimed as native-vendor decoding equivalence. Google keeps
its provider-native/reference `max_output_tokens` contract while the gateway wire
field is explicitly recorded as `max_tokens`.

## Zero-generation preflight

From `G:\ECIR2027`:

```powershell
python scripts/check_zhizengzeng_gateway.py --env-file .env --strict
```

This makes no generation request. It verifies all six exact model IDs and records
current gateway balance without printing the API key.

## Hosted FairSynth plan

First materialize the canonical 360-user synthetic freeze if it is not already
present:

```powershell
python scripts/build_fairsynth360.py `
  --output-dir data/frozen/fairsynth360 `
  --users 360 `
  --candidate-set-size 30 `
  --max-history-items 8 `
  --seed 1729
```

Then compile the default budget-sized hosted plan:

```powershell
python scripts/plan_hosted_fairsynth.py `
  --freeze-root data/frozen `
  --output-dir results/plans/hosted-fairsynth-budget-v1 `
  --users 120 `
  --repetitions 3 `
  --seed 1729
```

The 120-user subset is selected deterministically and exactly balanced across
meaningless identity groups A/B/C (40/40/40). With six registered FairSynth
conditions, six hosted model families, and three generations, the complete plan
contains **12,960 immutable cells**. The budget guard may stop before full
coverage; any incomplete coverage is reported explicitly.

## Execution

The runner is dry-run by default:

```powershell
python scripts/run_hosted_budgeted.py `
  --plan-dir results/plans/hosted-fairsynth-budget-v1 `
  --freeze-root data/frozen `
  --output-jsonl results/runs/hosted-fairsynth-budget-v1.jsonl
```

For the first paid canary execute only 12 cells:

```powershell
python scripts/run_hosted_budgeted.py `
  --plan-dir results/plans/hosted-fairsynth-budget-v1 `
  --freeze-root data/frozen `
  --output-jsonl results/runs/hosted-fairsynth-budget-v1.jsonl `
  --ledger results/budget/hosted_zzz_v1.json `
  --target-rmb 200 `
  --hard-cap-rmb 250 `
  --request-reserve-rmb 2 `
  --max-cells 12 `
  --execute
```

Inspect the 12 persisted rows, invalid-output accounting, returned model strings,
usage metadata, and budget ledger. Then resume the same plan without
`--max-cells`; completed `planned_cell_id` values are skipped exactly.

## Paper synchronization

FairSynth numerical paper content is generated only from audited artifacts:

```text
hosted JSONL
  -> scripts/audit_run_log.py
  -> scripts/analyze_fairsynth360.py
  -> scripts/render_fairsynth_paper_results.py
  -> paper/generated/fairsynth_hosted_table.tex
  -> paper/results_contract_table.tex (automatic include)
  -> Overleaf bundle
```

The whole post-run path is wrapped by:

```powershell
python scripts/finalize_hosted_fairsynth.py
```

That command audits the run, executes frozen FairSynth inference, renders the
six-model synthetic sanity table, runs paper preflights, rebuilds the anonymous
Overleaf ZIP, and writes a cryptographic paper-sync manifest. Do not type a
result value into the manuscript by hand.
