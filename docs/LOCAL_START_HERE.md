# FairEval — Start Here Before Real Execution

This is the shortest safe path from `G:\ECIR2027` to the first hosted or GPU canary. Everything through the pre-execution seal is zero-generation work: no hosted model call and no local model weight loading.

## 1. Pull the branch and keep it clean

```powershell
cd G:\ECIR2027
git fetch origin
git checkout ecir-2027-redesign
git pull --ff-only origin ecir-2027-redesign
$SHA = (git rev-parse HEAD).Trim()
git status --short
```

Do not continue to a scientific seal if tracked prompts/configs/code/tests/paper contracts are dirty.

## 2. Install the zero-call environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,analysis,providers]"
```

For the later AI Galaxy white-box run, install the CUDA-matched PyTorch environment plus `requirements-local-gpu.txt` on the GPU machine.

## 3. Build the canonical pre-execution seal

```powershell
python scripts\build_preexecution_seal.py `
  --output-dir results\preexecution\seal-v1
```

This one zero-call command builds/verifies the canonical 360-user FairSynth freeze, the 12,960-cell local white-box plan, the 12,960-cell hosted FairSynth plan, result contracts, paper figures, pre-result Overleaf ZIP, scientific source hashes, and unresolved real-dataset blockers.

The seal must state:

```text
hosted_api_generation_calls_made = 0
local_model_weights_loaded = false
empirical_results_seen_or_inserted = false
whitebox_core planned_cells = 12960
hosted_fairsynth planned_cells = 12960
scientific_worktree_clean = true
```

Real hosted/GPU launchers verify the seal again, including the exact Git SHA, scientific source hashes, plan hash, and planned-cell count.

## 4. Hosted gateway preflight

Keep the real key only in local `.env`:

```dotenv
FAIREVAL_HOSTED_GATEWAY=zhizengzeng
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
ZZZ_API_KEY=<secret>
```

Then:

```powershell
python scripts\check_zhizengzeng_gateway.py --env-file .env --strict
```

This makes no generation call. All six exact frozen model IDs from `configs/models.yaml` must be available before paid execution. There is no remaining separate Llama-host configuration in the current gateway design.

Hosted spend uses a 200 RMB normal stop, a 250 RMB client-side emergency stop threshold, and a 2 RMB reserve with gateway balance reconciliation. The 250 RMB value is not claimed as an atomic provider-side spending cap.

## 5. White-box GPU preflight

On AI Galaxy/Linux after cloning the exact sealed revision:

```bash
pip install -r requirements-local-gpu.txt
pip install -e .
nvidia-smi
python scripts/check_local_gpu.py --strict
```

The exact Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct revisions are already frozen in `configs/local_models.yaml`.

Dry-run two Qwen cells without loading model weights:

```bash
python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family qwen25_local \
  --output-jsonl results/runs/whitebox-qwen25-v1.jsonl \
  --max-cells 2
```

Only after reviewing that summary should the same command receive the seal path, exact Git SHA, and `--execute`.

## 6. Keep smoke data separate

A tiny 12-user smoke must use a separate path such as:

```text
data/frozen-smoke/fairsynth360
```

Never overwrite `data/frozen/fairsynth360`, which is reserved for the canonical 360-user execution freeze. CI enforces this separation.

## 7. Real-world data remains a hard gate

Use `configs/dataset_acquisition.yaml`, `configs/dataset_releases.yaml`, `scripts/check_raw_dataset_layout.py`, and `scripts/hash_raw_release.py` to acquire/license-review/hash the exact six third-party releases. Do not invent checksums or silently substitute releases.

The project-owned FairSynth hosted/local campaigns can proceed before those real-world freezes. Real-world RQ1–RQ4 cannot.

## 8. Paper updates are artifact-only

Hosted FairSynth uses `scripts/finalize_hosted_fairsynth.py`. Completed two-model white-box evidence uses `scripts/finalize_whitebox.py`. These paths audit the frozen outputs and generate paper artifacts automatically.

If evidence is not available, the paper remains pending. Do not manually type empirical numbers into `paper/main.tex` or result tables.

For the full operational procedure, use `docs/EXECUTION_RUNBOOK_2026-09-17.md`.
