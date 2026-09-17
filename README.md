# FairEval

FairEval is the ECIR 2027 preference-conditioned evaluation framework for fairness in LLM-based recommendation with personality awareness.

## Current execution checkpoint

The active research branch is `ecir-2027-redesign`. The hosted black-box phase now runs through the Zhizengzeng OpenAI-compatible gateway with a frozen **200 RMB planning target** and **250 RMB hard ceiling**. Hosted execution is dry-run by default and reconciles spend from the gateway account balance before and after every persisted experiment cell.

The current hosted panel is frozen in `configs/models.yaml`:

- OpenAI: `gpt-5.6-terra`
- Anthropic: `claude-sonnet-5`
- Google: `gemini-3.8-flash`
- DeepSeek: `deepseek-v4.1-flash`
- Alibaba Qwen: `qwen3.8-2.4t-a95b`
- Meta: `llama-4-maverick`

A live `GET /v1/models` preflight on the experiment key is still required before paid execution. If an exact frozen ID is missing, create a new experiment version rather than silently substituting a model.

## Hosted-only setup

Copy `.env.example` to `.env` and set only the gateway secret for the current hosted phase:

```dotenv
FAIREVAL_HOSTED_GATEWAY=zhizengzeng
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
ZZZ_API_KEY=<secret>
```

Do not commit `.env` or any real key.

Install the project and provider dependency:

```bash
python -m pip install -e ".[analysis,dev,providers]"
```

Run repository checks before any paid call:

```bash
python scripts/check_config_consistency.py
python scripts/check_pilot_readiness.py
pytest -q
```

`check_pilot_readiness.py --strict` intentionally remains blocked until all six third-party dataset releases are locally frozen and hashed.

## 200/250 RMB hosted budget guard

Budget policy is versioned in `configs/hosted_budget.yaml` and documented in `docs/HOSTED_API_BUDGET_PROTOCOL.md`.

The runner is a zero-call dry run unless `--execute` is supplied:

```bash
python scripts/run_hosted_budgeted.py \
  --plan-dir results/plans/hosted-core-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/runs/hosted-core-v1.jsonl
```

The real run uses the same immutable plan plus the non-bypassable budget ledger:

```bash
python scripts/run_hosted_budgeted.py \
  --plan-dir results/plans/hosted-core-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/runs/hosted-core-v1.jsonl \
  --ledger results/budget/hosted_zzz_v1.json \
  --target-rmb 200 \
  --hard-cap-rmb 250 \
  --request-reserve-rmb 2 \
  --execute
```

For the first paid canary, add `--max-cells 12`; audit those rows before scaling. The ledger records the experiment's initial available balance and reconciles spend as balance movement. A new request is not launched once remaining budget is at or below the frozen reserve.

## Dataset and experiment safety

FairEval does not infer protected attributes from names, free text, embeddings, ZIP codes, or model guesses. Third-party raw releases must be locally acquired under their documented terms, hashed, and frozen before execution. Project-generated FairSynth-360 remains a separately reported controlled sanity benchmark rather than evidence about human demographic fairness.

Run plans are immutable JSONL artifacts. Their file hash, individual cell IDs, model IDs, conditions, candidate order, generation settings, and source-config hashes are verified before execution. Persistent invalid model output is retained as an experimental outcome rather than regenerated until valid.

## Results and paper synchronization

Empirical numbers are never typed into the paper by hand. The paper-facing chain is:

```text
frozen run JSONL
→ run-log audit
→ RQ analysis artifacts
→ result-table renderer
→ result-figure renderer
→ paper result renderer
→ Overleaf bundle
```

Relevant scripts include:

```text
scripts/audit_run_log.py
scripts/analyze_core.py
scripts/analyze_rq2_traits.py
scripts/analyze_rq3.py
scripts/analyze_rq4.py
scripts/render_result_tables.py
scripts/render_result_tables_lncs.py
scripts/build_result_figures.py
scripts/render_paper_results.py
scripts/build_overleaf_bundle.py
```

Until audited artifacts exist, corresponding paper cells remain explicitly pending. This avoids hallucinated or manually transcribed numerical results.

## Local/open-weight transparency track

The Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct local track remains available for later reproducibility/white-box analysis, but it is **not required for the current hosted-only API phase**. Its configuration stays in `configs/local_models.yaml` and `requirements-local-gpu.txt`.

## Core scientific principle

A recommendation change is sensitivity, not automatically unfairness. FairEval evaluates the held-out consequence of matched context interventions, separates measured personality value from shuffled/counterfactual controls, and reports robustness across registered prompts, cues, candidate order, stochastic generations, models, and domains.
