# FairEval ECIR 2027 Local/Execution Runbook

This document is a compact local entrypoint. The authoritative detailed procedure is `docs/EXECUTION_RUNBOOK_2026-09-17.md`. The current study has two separately reported evidence lanes: six hosted families through Zhizengzeng and two frozen local open-weight models on a large CUDA GPU.

## 1. Pull one clean revision

From `G:\ECIR2027`:

```powershell
cd G:\ECIR2027
git fetch origin
git checkout ecir-2027-redesign
git pull --ff-only origin ecir-2027-redesign
$SHA = (git rev-parse HEAD).Trim()
git status --short
```

Do not start a real run from a dirty scientific worktree. Once the first experimental row is persisted, do not change executable source inside that experiment version.

## 2. Install the zero-call environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,analysis,providers]"
```

Never commit `.env` or any API key.

## 3. Hosted gateway environment

The black-box lane uses one gateway configuration, not six independent provider credentials:

```dotenv
FAIREVAL_HOSTED_GATEWAY=zhizengzeng
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
ZZZ_API_KEY=<secret>
```

The exact six model IDs are frozen in `configs/models.yaml`. A live model-list preflight is mandatory before paid execution; nearby model names are not substitutes.

The hosted experiment normally stops around 200 RMB. A separate 250 RMB client-side emergency stop threshold and 2 RMB reserve provide additional protection. Balance is reconciled before/after persisted cells. This is not represented as an atomic provider-side spending cap.

## 4. Build the zero-call scientific seal

Before hosted generation or local model loading:

```powershell
python scripts\build_preexecution_seal.py `
  --output-dir results\preexecution\seal-v1
```

The seal performs the deterministic pre-experiment work in one path: canonical FairSynth generation, hosted/local plan construction, configuration and paper checks, pre-result Overleaf bundle generation, source-file hashing, Git-SHA capture, and real-dataset blocker reporting.

Expected core geometry in the seal:

```text
white-box FairSynth core: 12,960 cells
hosted FairSynth plan:    12,960 cells
hosted generation calls: 0
local model weights:      not loaded
empirical paper results:  none
```

Both real runners later verify the exact sealed commit, scientific source hashes, plan hash, and cell count.

## 5. Hosted dry run and canary

First run the no-generation gateway check:

```powershell
python scripts\check_zhizengzeng_gateway.py --env-file .env --strict
```

Then dry-run the sealed hosted plan:

```powershell
python scripts\run_hosted_budgeted.py `
  --plan-dir results\plans\hosted-fairsynth-budget-v1 `
  --freeze-root data\frozen `
  --output-jsonl results\runs\hosted-fairsynth-budget-v1.jsonl `
  --ledger results\budget\hosted_zzz_v1.json `
  --env-file .env `
  --max-cells 12
```

Paid execution additionally requires the exact current `$SHA`, the seal path, and `--execute`. Start with one cell per hosted family as documented in the authoritative execution runbook. Exit code `10` means the budget policy intentionally stopped the run; do not bypass it.

## 6. Large-GPU white-box lane

Use a Linux CUDA environment on AI Galaxy or equivalent. Install the CUDA-matched PyTorch build, then:

```bash
pip install -r requirements-local-gpu.txt
pip install -e .
python scripts/check_local_gpu.py --strict
```

The exact Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct revisions are already frozen in `configs/local_models.yaml`. The canonical FairSynth plan is 12,960 generations. Execute one family at a time with `scripts/run_whitebox_family.py`; real inference requires the same pre-execution seal and exact Git SHA.

Do not quantize the canonical run merely to fit a smaller GPU. Use suitable hardware or create a separately versioned protocol.

## 7. Third-party real datasets

The six real-world datasets are intentionally not considered execution-ready until their exact upstream releases are present locally and `configs/dataset_releases.yaml` records:

- exact release identity;
- deterministic SHA-256 freeze;
- reviewed license/terms;
- valid local raw path.

Do not invent hashes or silently substitute a newer/different release. Real-world RQ1–RQ4 starts only after those locks pass.

## 8. Artifact-only paper updates

Numerical paper content follows this rule:

```text
frozen run JSONL
→ run-log audit
→ registered analysis
→ generated LaTeX/PDF artifact
→ paper/generated/
→ anonymous Overleaf bundle
```

Hosted FairSynth finalization uses `scripts/finalize_hosted_fairsynth.py`. Full white-box finalization uses `scripts/finalize_whitebox.py`. Real-world RQ1–RQ4 uses the result-table/figure pipeline documented in `docs/RESULT_TABLES.md`.

If a required artifact is missing, the manuscript stays explicitly pending. Never type a result value into the manuscript by hand.

## 9. Stop rather than improvise

Stop the current experiment version if the frozen model ID disappears, source/seal verification fails, Git SHA changes, CUDA/BF16 preflight fails, a run-log audit fails, a real dataset lock is incomplete, or the client-side hosted budget policy stops execution. Versioned incomplete evidence is preferable to an untracked substitution.
