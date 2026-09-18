# FairEval ECIR 2027 — frozen execution runbook

This document is the operational handoff from the repository-ready checkpoint to real hosted and GPU execution. It is deliberately conservative about provenance: **pull once, build the zero-call pre-execution seal, record `git rev-parse HEAD`, and do not change the checkout after the first persisted experimental row**. Both hosted and local launchers reject a claimed code SHA that differs from the checked-out Git `HEAD` and reject an execution plan that does not match the seal.

The two evidence lanes are separate:

- **hosted black-box:** six frozen families through the Zhizengzeng OpenAI-compatible gateway, with a 200 RMB normal-stop target and a 250 RMB client-side emergency stop threshold;
- **local white-box:** two exact Hugging Face revisions on a large CUDA GPU, with no token-budget constraint and a canonical 12,960-generation FairSynth campaign before real-dataset replication.

FairEval does not claim the 250 RMB value is an atomic provider-side spending cap. The hosted client normally stops around 200 RMB and retains roughly 50 RMB of operational headroom while reconciling balance before and after every persisted cell.

No numerical paper result is typed by hand. Audited artifacts are the only path into `paper/generated/`.

## 1. Freeze the exact code checkpoint on Windows

From `G:\ECIR2027`:

```powershell
cd G:\ECIR2027

git fetch origin
git checkout ecir-2027-redesign
git pull --ff-only origin ecir-2027-redesign

$SHA = (git rev-parse HEAD).Trim()
Write-Host "FAIREVAL_EXECUTION_SHA=$SHA"
git status --short
```

`git status --short` should be empty before the seal and before the first real execution. If it is not empty, preserve/review the local changes rather than resetting blindly.

Install the canonical Windows/CPU/hosted environment through the repository-level requirements entry point:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` is part of the sealed scientific specification. The separate `requirements-local-gpu.txt` includes this base environment and adds the white-box Transformers stack; CUDA-matched PyTorch remains machine-specific and must be installed for the selected GPU image.

## 2. Build the zero-call scientific seal

This is mandatory before either evidence lane:

```powershell
python scripts\build_preexecution_seal.py `
  --output-dir results\preexecution\seal-v1
```

The seal performs no hosted generation and loads no local model weights. It regenerates the canonical FairSynth freeze, compiles both deterministic FairSynth plans, renders pre-result table/figure contracts, runs configuration/manuscript preflights, hashes the tracked scientific specification, records unresolved third-party dataset blockers, and builds a pre-result anonymous Overleaf ZIP.

Verify the seal exists:

```powershell
Get-Content results\preexecution\seal-v1\PREEXECUTION_SEAL.md
```

After sealing, `$SHA = (git rev-parse HEAD).Trim()` must remain exactly the seal's commit. If executable scientific source changes, regenerate the seal and use a new run/version rather than continuing an old run log.

After the first hosted or local result row is persisted, **do not `git pull`, checkout another commit, or edit executable source in that run directory**.

## 3. Hosted Zhizengzeng environment — secrets stay local

Merge the following into the local `.env`; never commit the real key:

```dotenv
FAIREVAL_HOSTED_GATEWAY=zhizengzeng
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
ZZZ_API_KEY=<YOUR_REAL_KEY>
```

Check only the live model list first; this makes no generation call:

```powershell
python scripts\check_zhizengzeng_gateway.py --env-file .env
```

Paid execution must not start unless all six exact IDs in `configs/models.yaml` are returned by the account's live `/v1/models` endpoint. Nearby model names are not substitutes.

## 4. Hosted dry run, six-family canary, then budgeted continuation

The seal already generated `results/plans/hosted-fairsynth-budget-v1`. Dry-run first; no generation call:

```powershell
python scripts\run_hosted_budgeted.py `
  --plan-dir results\plans\hosted-fairsynth-budget-v1 `
  --freeze-root data\frozen `
  --output-jsonl results\runs\hosted-fairsynth-budget-v1.jsonl `
  --ledger results\budget\hosted_zzz_v1.json `
  --preexecution-seal results\preexecution\seal-v1\PREEXECUTION_SEAL.json `
  --env-file .env `
  --target-rmb 200 `
  --hard-cap-rmb 250 `
  --request-reserve-rmb 2 `
  --max-cells 12
```

For the first paid validation, execute **one cell from each family** against the same output log and budget ledger. Re-evaluate `$SHA = (git rev-parse HEAD).Trim()` immediately before these commands; it must match the seal.

```powershell
$SHA = (git rev-parse HEAD).Trim()
$families = @("openai", "anthropic", "google", "deepseek", "qwen", "meta")
foreach ($family in $families) {
  python scripts\run_hosted_budgeted.py `
    --plan-dir results\plans\hosted-fairsynth-budget-v1 `
    --freeze-root data\frozen `
    --output-jsonl results\runs\hosted-fairsynth-budget-v1.jsonl `
    --ledger results\budget\hosted_zzz_v1.json `
    --preexecution-seal results\preexecution\seal-v1\PREEXECUTION_SEAL.json `
    --env-file .env `
    --family $family `
    --max-cells 1 `
    --target-rmb 200 `
    --hard-cap-rmb 250 `
    --request-reserve-rmb 2 `
    --code-commit-sha $SHA `
    --execute
  if ($LASTEXITCODE -notin @(0,10)) { throw "Hosted canary failed for $family" }
  if ($LASTEXITCODE -eq 10) { break }
}
```

Exit code `10` means the client-side budget policy intentionally stopped execution. Do not work around it.

If the six-family canary is clean, continue in deterministic batches using the **same log, ledger, seal, and commit**:

```powershell
$SHA = (git rev-parse HEAD).Trim()
python scripts\run_hosted_budgeted.py `
  --plan-dir results\plans\hosted-fairsynth-budget-v1 `
  --freeze-root data\frozen `
  --output-jsonl results\runs\hosted-fairsynth-budget-v1.jsonl `
  --ledger results\budget\hosted_zzz_v1.json `
  --preexecution-seal results\preexecution\seal-v1\PREEXECUTION_SEAL.json `
  --env-file .env `
  --max-cells 250 `
  --target-rmb 200 `
  --hard-cap-rmb 250 `
  --request-reserve-rmb 2 `
  --code-commit-sha $SHA `
  --execute
```

Repeat that batch command while the budget guard permits. The ledger reconciles experiment spend from gateway balance movement before/after persisted cells; do not estimate remaining allowance manually, create a new ledger to evade the frozen policy, or describe the 250 RMB threshold as provider-enforced atomic billing protection.

When hosted execution is complete/stopped by policy, update the paper only through:

```powershell
python scripts\finalize_hosted_fairsynth.py
```

That path audits the run, runs the frozen FairSynth analysis, renders the artifact-derived LaTeX results table, renders `paper/figures/fairsynth_hosted_effects.pdf` directly from the same inference artifact, and rebuilds the anonymous Overleaf ZIP. No result number or plotted coordinate is copied into the paper by hand. If coverage is insufficient for a particular contrast, report that coverage limitation rather than fabricating a complete panel.

## 5. AI Galaxy large-GPU white-box environment

Use a Linux CUDA/PyTorch image. Prefer a 32–48+ GiB NVIDIA GPU; A100/H100/L40S/RTX-6000-class hardware is suitable. The two models run sequentially, so multi-model co-residency is unnecessary.

Clone/pull the exact **same sealed commit** used to create the seal. Copy the seal directory and generated plan directory with the project workspace, or rebuild the seal on the identical clean checkout before any inference.

```bash
git clone -b ecir-2027-redesign https://github.com/Rocky5502/-FairEval.git FairEval
cd FairEval
git pull --ff-only origin ecir-2027-redesign
SHA=$(git rev-parse HEAD)
echo "FAIREVAL_EXECUTION_SHA=$SHA"
git status --short
```

Do not start if the checkout is dirty. Build the zero-call seal on this exact checkout if it is not already present:

```bash
python scripts/build_preexecution_seal.py \
  --output-dir results/preexecution/seal-v1
```

Verify the GPU and environment:

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('cuda_runtime=', torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('gpu=', torch.cuda.get_device_name(0))
    print('vram_gib=', p.total_memory / 1024**3)
PY
```

If the image already has CUDA-matched PyTorch, keep it and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# If venv isolation hides the image's torch, install the CUDA-matched torch build first.
pip install -r requirements-local-gpu.txt
pip install -e .
python scripts/check_local_gpu.py --strict
```

Do not replace the frozen BF16 protocol with quantization just to fit a smaller GPU. Use a larger GPU or version a new protocol.

## 6. Canonical full white-box FairSynth campaign

The pre-execution seal builds the canonical plan at:

```text
results/plans/whitebox-full-v1/core
```

Its manifest must report exactly **12,960 FairSynth cells**:

```text
360 users x 6 registered conditions x 3 seeded repetitions x 2 frozen models
```

Real-world datasets are intentionally absent from this first campaign while their exact licensed releases/hashes remain unfrozen. Do not bypass that gate.

## 7. White-box canary and full resume-safe execution

Dry-run Qwen first:

```bash
python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family qwen25_local \
  --output-jsonl results/runs/whitebox-qwen25-v1.jsonl \
  --preexecution-seal results/preexecution/seal-v1/PREEXECUTION_SEAL.json \
  --max-cells 2
```

Execute only two canary cells:

```bash
SHA=$(git rev-parse HEAD)
python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family qwen25_local \
  --output-jsonl results/runs/whitebox-qwen25-v1.jsonl \
  --preexecution-seal results/preexecution/seal-v1/PREEXECUTION_SEAL.json \
  --max-cells 2 \
  --code-commit-sha "$SHA" \
  --execute
```

Repeat the two-cell canary for Phi using its own output log. Audit/inspect those persisted rows before scaling.

Then run both families in resume-safe batches. A large GPU can use a larger `--max-cells` batch (for example 250–1000); batching changes operational checkpoint frequency, **not** the frozen plan.

```bash
SHA=$(git rev-parse HEAD)

python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family qwen25_local \
  --output-jsonl results/runs/whitebox-qwen25-v1.jsonl \
  --preexecution-seal results/preexecution/seal-v1/PREEXECUTION_SEAL.json \
  --max-cells 500 \
  --code-commit-sha "$SHA" \
  --execute

python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family phi35_local \
  --output-jsonl results/runs/whitebox-phi35-v1.jsonl \
  --preexecution-seal results/preexecution/seal-v1/PREEXECUTION_SEAL.json \
  --max-cells 500 \
  --code-commit-sha "$SHA" \
  --execute
```

Repeat until both families have zero pending core cells. The runner refuses a `--code-commit-sha` that differs from the current checkout, refuses a seal that differs from the checkout/plan, and the finalizer rejects mixed/partial coverage.

## 8. Finalize the full white-box evidence directly into the paper

After both family logs completely cover the core plan:

```bash
python scripts/finalize_whitebox.py \
  --input-jsonl results/runs/whitebox-qwen25-v1.jsonl \
  --input-jsonl results/runs/whitebox-phi35-v1.jsonl \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --output-dir results/analysis/whitebox-full-v1 \
  --paper-table paper/generated/whitebox_summary_table.tex \
  --overleaf-output dist/FairEval_ECIR2027_Overleaf.zip
```

The finalizer:

1. audits each family log and frozen seed contract;
2. refuses duplicates, missing plan cells, or non-local families;
3. merges in immutable plan order and re-audits the merged artifact;
4. scores recommendation utility and derives the declared auxiliary white-box summaries;
5. writes the generated LaTeX table;
6. rebuilds the anonymous Overleaf ZIP including the generated table.

This is the only canonical path from the full local run to paper numbers.

## 9. Real-world RQ1/RQ2/RQ3/RQ4 after exact dataset freezes

The external datasets remain a hard data-provenance gate. Once all exact release locks are complete, create a **new study/seal version** rather than modifying the executed FairSynth seal/plan in place:

```bash
python scripts/build_whitebox_campaign.py \
  --freeze-root data/frozen \
  --output-root results/plans/whitebox-real-v1 \
  --include-real-world

python scripts/plan_whitebox_rq3.py \
  --freeze-root data/frozen \
  --output-dir results/plans/whitebox-real-v1/rq3
```

The pre-execution seal format currently seals the canonical FairSynth hosted/local plans. Before real-world execution, version/extend the seal to include those exact real-world core/RQ3 plans; do not reuse the FairSynth-only seal as if it covered new plans.

Execute the real-world core and RQ3 plan with `scripts/run_whitebox_family.py`, one family at a time, using the same frozen checkout. Derive RQ4 identity-irrelevance prompting from the executed local core via `scripts/build_rq4_prompt_plan.py`; PAIR remains post-processing with validation-only operating-point selection.

Hosted real-world RQ1/RQ2/RQ3/RQ4 uses the same dataset freezes but remains constrained by the existing 200 RMB normal-stop policy and 250 RMB emergency threshold. Local GPU coverage may be complete even if hosted coverage is budget-truncated; report the two strata honestly rather than implying equal sample sizes.

## 10. Stop conditions

Stop rather than improvise if any of the following occurs:

- the live gateway omits an exact frozen hosted model ID;
- the hosted ledger reaches its normal stop or emergency threshold;
- the pre-execution seal no longer matches Git `HEAD` or the plan SHA;
- `git rev-parse HEAD` changes after execution starts;
- CUDA/BF16 preflight fails on the white-box machine;
- a local model revision cannot be resolved exactly;
- a run-log audit fails;
- a real-world dataset release/hash/license lock is incomplete;
- generated paper artifacts cannot be traced to an audited run/analysis manifest.

A stopped, explicitly incomplete experiment is scientifically preferable to an untracked substitution or manually repaired paper result.
