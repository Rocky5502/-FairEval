# FairEval — local execution start here

This is the shortest safe path from a fresh `ecir-2027-redesign` checkout to the first real local-model calls. Commands before the final `--execute` example make **zero provider API calls**.

## 1. Environment

Windows PowerShell:

```powershell
git checkout ecir-2027-redesign
git pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA-matched PyTorch wheel appropriate for the workstation first. Then install the complete Week-1 environment:

```powershell
python -m pip install -r requirements-week1.txt
python -m pip install -e .
```

Run the zero-call repository bootstrap:

```powershell
python scripts/week1_bootstrap.py
python scripts/check_local_gpu.py
```

Do not begin local inference until `check_local_gpu.py --strict` passes. The exact Qwen2.5 and Phi-3.5 repository commits are already frozen in `configs/local_models.yaml`.

## 2. Freeze the full project-owned FairSynth benchmark

FairSynth does not require third-party raw files; `--raw-dir .` is an explicit unused placeholder for the shared freeze interface.

```powershell
faireval prepare `
  --dataset fairsynth360 `
  --raw-dir . `
  --output-dir data/frozen/fairsynth360 `
  --users 360 `
  --candidate-set-size 30 `
  --max-history-items 8 `
  --seed 1729

faireval verify-freeze --output-dir data/frozen/fairsynth360
```

For a tiny pipeline smoke, substitute `--users 12` and use a separate output directory such as `data/frozen-smoke/fairsynth360`; never overwrite the full freeze with a smoke freeze.

## 3. Compile the immutable Qwen/Phi plan

Before the six third-party datasets exist, compile a FairSynth-only plan:

```powershell
faireval plan-local `
  --freeze-root data/frozen `
  --output-dir results/plans/local-fairsynth-v1 `
  --fairsynth-users 360 `
  --repetitions 3 `
  --no-real-world
```

Review `results/plans/local-fairsynth-v1/plan_manifest.json`. Do not edit the JSONL or manifest by hand after compilation.

## 4. Prove the execution path without loading a model

```powershell
faireval execute-plan `
  --plan-dir results/plans/local-fairsynth-v1 `
  --freeze-root data/frozen `
  --output-jsonl results/raw/local-fairsynth-v1.jsonl `
  --family phi35_local `
  --max-cells 2
```

The summary must report `api_calls_made: 0`. This dry run does not construct the Transformers provider or load weights.

You can also run the CI-equivalent end-to-end smoke locally:

```powershell
python scripts/zero_call_local_smoke.py --output results/smoke/local_zero_call_v1.json
```

## 5. First real GPU smoke

Only after the GPU preflight and dry run are green, capture the exact code SHA and execute **two Phi cells only**:

```powershell
$sha = git rev-parse HEAD
faireval execute-plan `
  --plan-dir results/plans/local-fairsynth-v1 `
  --freeze-root data/frozen `
  --output-jsonl results/raw/local-fairsynth-v1.jsonl `
  --family phi35_local `
  --max-cells 2 `
  --code-commit-sha $sha `
  --execute
```

Audit those rows before expanding the run. Then repeat the same tiny smoke for `qwen25_local`. Do not delete malformed outputs; persistent invalidity is an experimental outcome.

## 6. External-data / hosted track

In parallel, use `configs/dataset_acquisition.yaml` and `configs/dataset_releases.yaml` to download, license-review, hash, and freeze the exact six real-world releases. The hosted core does not start until:

```powershell
python scripts/check_pilot_readiness.py --strict
```

passes. The remaining hosted-specific blocker is the exact Llama host/endpoint/served revision plus provider credentials.

## 7. Daily checkpoint

At the end of every work session keep these immutable artifacts:

- `git rev-parse HEAD`;
- environment manifest / `pip freeze`;
- dataset freeze manifests;
- run-plan manifest(s);
- raw JSONL without manual edits;
- audit/analyzer manifests;
- generated paper tables/figures only from frozen artifacts.

The longer Day-1-to-Day-7 schedule is in `docs/WEEK_ONE_EXECUTION.md`.
