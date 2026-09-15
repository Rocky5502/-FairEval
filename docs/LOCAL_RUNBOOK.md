# FairEval Local Runbook

This runbook is the intended path from a fresh clone to a frozen ECIR 2027 run. It is deliberately conservative: preflight and dry-run steps make **zero provider API calls**.

## 1. Create the Python environment

### Windows PowerShell

```powershell
cd G:\path\to\-FairEval
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### WSL / Linux

```bash
cd /path/to/-FairEval
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` pins the provider SDK versions used by the experiment and bounds the scientific stack. Before a frozen pilot/confirmatory release, archive `python -m pip freeze` alongside the run manifest.

## 2. Configure credentials without committing them

Use `.env.example` only as a checklist. The code reads provider credentials from the process environment; it does **not** require committing or auto-loading a `.env` file.

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "..."
$env:ANTHROPIC_API_KEY = "..."
$env:GEMINI_API_KEY = "..."
$env:DEEPSEEK_API_KEY = "..."
$env:DASHSCOPE_API_KEY = "..."
$env:QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Freeze one Llama host before the pilot:
$env:LLAMA_PROVIDER_API_KEY = "..."
$env:LLAMA_BASE_URL = "..."
$env:LLAMA_PROVIDER_NAME = "..."
```

Never paste real keys into YAML, logs, screenshots, issues, or commits.

## 3. Zero-call preflight

Run these before touching any paid provider:

```bash
python scripts/check_config_consistency.py
python scripts/check_environment.py
pytest -q
python scripts/build_paper_figures.py
python scripts/check_paper_source.py
python scripts/write_environment_manifest.py --output results/environment_manifest.json
```

Expected properties:

- six executable dataset IDs agree across adapters and configs;
- MIND stays preference-only in the frozen core plan;
- primary prompt/cue/model-generation settings agree across manifests;
- provider credential presence is reported without printing secrets;
- the three methods figures regenerate as vector PDFs;
- every cited BibTeX key and LaTeX input resolves;
- all unit tests pass.

## 4. Inspect dataset adapters before downloading/running

```bash
faireval dataset-card --dataset personality2018
faireval dataset-card --dataset music_master_bfi2
faireval dataset-card --dataset reasoner
faireval dataset-card --dataset movielens_1m
faireval dataset-card --dataset lastfm_1k
faireval dataset-card --dataset mind
```

Confirm the upstream license/terms and raw schema for the exact release you will use. Do not silently substitute a different release after the pilot freeze.

## 5. Freeze deterministic benchmark instances

Example:

```bash
faireval prepare \
  --dataset movielens_1m \
  --raw-dir data/raw/movielens_1m \
  --output-dir data/frozen/movielens_1m \
  --users 20 \
  --candidate-set-size 50 \
  --max-history-items 20 \
  --seed 1729

faireval verify-freeze --output-dir data/frozen/movielens_1m
```

Repeat for all six datasets. The pilot default is 20 users per dataset. The final confirmatory N is frozen only after the variance/invalid-output pilot, following `configs/study_design.yaml`.

## 6. Compile the immutable core run plan

After **all six** frozen dataset directories exist:

```bash
faireval plan-core \
  --freeze-root data/frozen \
  --output-dir results/plans/core-v1 \
  --counterfactuals configs/counterfactuals.yaml \
  --models configs/models.yaml \
  --seed 1729
```

The command writes `run_plan.jsonl` plus `plan_manifest.json`, including dataset hashes and a plan hash. Do not hand-edit either file after compilation.

## 7. Dry-run before any API call

```bash
faireval execute-plan \
  --plan-dir results/plans/core-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/raw/core-v1.jsonl
```

Without `--execute`, this is a **zero-call dry run**. Review the planned cell count, family filters, credentials, provider endpoints, and expected cost before proceeding.

To inspect only one family without calling it:

```bash
faireval execute-plan \
  --plan-dir results/plans/core-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/raw/core-v1.jsonl \
  --family openai \
  --max-cells 5
```

## 8. Execute only after the freeze is approved

Capture the exact code commit first.

PowerShell:

```powershell
$sha = git rev-parse HEAD
faireval execute-plan `
  --plan-dir results/plans/core-v1 `
  --freeze-root data/frozen `
  --output-jsonl results/raw/core-v1.jsonl `
  --code-commit-sha $sha `
  --execute
```

WSL / Linux:

```bash
SHA="$(git rev-parse HEAD)"
faireval execute-plan \
  --plan-dir results/plans/core-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/raw/core-v1.jsonl \
  --code-commit-sha "$SHA" \
  --execute
```

The executor resumes by immutable `planned_cell_id`. An invalid model output still completes its planned cell because invalidity is an experimental outcome; do not rerun until valid.

## 9. Before promoting pilot to confirmatory

Do **not** use the pilot treatment-effect mean to chase significance. Use the pilot only for variance, invalid-output rate, runtime, and cost. Then freeze:

- final N and user split hashes;
- candidate sets and order;
- prompt template/cue suite;
- provider endpoints/regions;
- exact model IDs and deliberation policies;
- metric/statistical definitions;
- analysis code commit.

If any of those change, create a new versioned plan rather than mutating the old one.

## 10. Paper integrity

The manuscript intentionally keeps numerical results as `TBD` until generated from frozen outputs. Regenerate methods figures with:

```bash
python scripts/build_paper_figures.py
```

Then run:

```bash
python scripts/check_paper_source.py
```

The comparison table is an evidence-audited **study-design coverage** matrix, not a claim that FairEval empirically outperforms prior work. Its source ledger is `docs/LITERATURE_COMPARISON_AUDIT.md`.
