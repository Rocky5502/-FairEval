# FairEval ECIR 2027 — frozen execution runbook

This document is the operational handoff from the repository-ready checkpoint to real hosted and GPU execution. It is deliberately conservative about provenance: **pull once, record `git rev-parse HEAD`, run CI/preflights, and do not change the checkout after the first persisted experimental row**. Both hosted and local launchers reject a claimed code SHA that differs from the checked-out Git `HEAD`.

The two evidence lanes are separate:

- **hosted black-box:** six frozen families through the Zhizengzeng OpenAI-compatible gateway, with a 200 RMB normal-stop target and a non-bypassable 250 RMB hard ceiling;
- **local white-box:** two exact Hugging Face revisions on a large CUDA GPU, with no token-budget constraint and a canonical 12,960-generation FairSynth campaign before real-dataset replication.

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

`git status --short` should be empty before the first real execution. If it is not empty, preserve/review the local changes rather than resetting blindly.

After the first hosted or local result row is persisted, **do not `git pull`, checkout another commit, or edit executable source in that run directory**. Version a new experiment instead.

## 2. Hosted Zhizengzeng environment — secrets stay local

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

## 3. Build/freeze FairSynth-360 locally without any model call

```powershell
python scripts\build_fairsynth360.py `
  --output-dir data\frozen\fairsynth360 `
  --users 360 `
  --candidate-set-size 30 `
  --max-history-items 8 `
  --seed 1729

python -m faireval.cli verify-freeze `
  --output-dir data\frozen\fairsynth360
```

Compile the hosted budget-sized controlled plan:

```powershell
python scripts\plan_hosted_fairsynth.py `
  --freeze-root data\frozen `
  --output-dir results\plans\hosted-fairsynth-budget-v1 `
  --users 120 `
  --repetitions 3 `
  --seed 1729
```

The plan is deterministic. Selected users are exactly A/B/C balanced and then deterministically interleaved. Within every condition the executor plan cycles all six model families and their repetitions before advancing, which makes a budget-truncated prefix more useful than a family-blocked schedule.

## 4. Hosted dry run, six-family canary, then budgeted continuation

Dry-run first; no generation call:

```powershell
python scripts\run_hosted_budgeted.py `
  --plan-dir results\plans\hosted-fairsynth-budget-v1 `
  --freeze-root data\frozen `
  --output-jsonl results\runs\hosted-fairsynth-budget-v1.jsonl `
  --ledger results\budget\hosted_zzz_v1.json `
  --env-file .env `
  --target-rmb 200 `
  --hard-cap-rmb 250 `
  --request-reserve-rmb 2 `
  --max-cells 12
```

For the first paid validation, execute **one cell from each family** against the same output log and budget ledger. Re-evaluate `$SHA = (git rev-parse HEAD).Trim()` immediately before these commands; it must remain unchanged.

```powershell
$SHA = (git rev-parse HEAD).Trim()
$families = @("openai", "anthropic", "google", "deepseek", "qwen", "meta")
foreach ($family in $families) {
  python scripts\run_hosted_budgeted.py `
    --plan-dir results\plans\hosted-fairsynth-budget-v1 `
    --freeze-root data\frozen `
    --output-jsonl results\runs\hosted-fairsynth-budget-v1.jsonl `
    --ledger results\budget\hosted_zzz_v1.json `
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

Exit code `10` means the budget guard intentionally stopped execution. Do not work around it.

If the six-family canary is clean, continue in deterministic batches using the **same log and ledger**:

```powershell
$SHA = (git rev-parse HEAD).Trim()
python scripts\run_hosted_budgeted.py `
  --plan-dir results\plans\hosted-fairsynth-budget-v1 `
  --freeze-root data\frozen `
  --output-jsonl results\runs\hosted-fairsynth-budget-v1.jsonl `
  --ledger results\budget\hosted_zzz_v1.json `
  --env-file .env `
  --max-cells 250 `
  --target-rmb 200 `
  --hard-cap-rmb 250 `
  --request-reserve-rmb 2 `
  --code-commit-sha $SHA `
  --execute
```

Repeat that batch command while the budget guard permits. The ledger reconciles actual experiment spend from gateway balance movement before/after persisted cells; do not estimate the remaining allowance manually and do not create a new ledger to evade the frozen experiment ceiling.

When hosted execution is complete/stopped by policy, update the paper only through:

```powershell
python scripts\finalize_hosted_fairsynth.py
```

That path audits the run and renders artifact-derived FairSynth content. If coverage is insufficient for a particular contrast, report that coverage limitation rather than fabricating a complete panel.

## 5. AI Galaxy large-GPU white-box environment

Use a Linux CUDA/PyTorch image. Prefer a 32–48+ GiB NVIDIA GPU; A100/H100/L40S/RTX-6000-class hardware is suitable. The two models run sequentially, so multi-model co-residency is unnecessary.

Clone/pull the same branch and record its exact SHA:

```bash
git clone -b ecir-2027-redesign https://github.com/Rocky5502/-FairEval.git FairEval
cd FairEval
git pull --ff-only origin ecir-2027-redesign
SHA=$(git rev-parse HEAD)
echo "FAIREVAL_EXECUTION_SHA=$SHA"
git status --short
```

Do not start if the checkout is dirty.

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

## 6. Build the canonical full white-box FairSynth campaign

The campaign builder performs no model calls:

```bash
python scripts/build_whitebox_campaign.py \
  --freeze-root data/frozen \
  --output-root results/plans/whitebox-full-v1
```

The core manifest must report exactly **12,960 FairSynth cells**:

```text
360 users x 6 registered conditions x 3 seeded repetitions x 2 frozen models
```

Real-world datasets are intentionally absent from this first campaign unless their exact licensed releases and hashes have already been frozen. Do not bypass that gate.

## 7. White-box canary and full resume-safe execution

Dry-run Qwen first:

```bash
python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family qwen25_local \
  --output-jsonl results/runs/whitebox-qwen25-v1.jsonl \
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
  --max-cells 500 \
  --code-commit-sha "$SHA" \
  --execute

python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family phi35_local \
  --output-jsonl results/runs/whitebox-phi35-v1.jsonl \
  --max-cells 500 \
  --code-commit-sha "$SHA" \
  --execute
```

Repeat until both families have zero pending core cells. The runner refuses a `--code-commit-sha` that differs from the current checkout, and the finalizer rejects mixed/partial coverage.

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

The external datasets remain a hard data-provenance gate. Once all exact release locks are complete, version a new full campaign (do not mutate an already executed plan):

```bash
python scripts/build_whitebox_campaign.py \
  --freeze-root data/frozen \
  --output-root results/plans/whitebox-real-v1 \
  --include-real-world

python scripts/plan_whitebox_rq3.py \
  --freeze-root data/frozen \
  --output-dir results/plans/whitebox-real-v1/rq3
```

Execute the real-world core and RQ3 plan with `scripts/run_whitebox_family.py`, one family at a time, using the same frozen checkout. Derive RQ4 identity-irrelevance prompting from the executed local core via `scripts/build_rq4_prompt_plan.py`; PAIR remains post-processing with validation-only operating-point selection.

Hosted real-world RQ1/RQ2/RQ3/RQ4 uses the same dataset freezes but remains constrained by the existing 200/250 RMB experiment budget. Local GPU coverage may be complete even if hosted coverage is budget-truncated; report the two strata honestly rather than implying equal sample sizes.

## 10. Stop conditions

Stop rather than improvise if any of the following occurs:

- the live gateway omits an exact frozen hosted model ID;
- the hosted ledger reaches its normal stop or emergency ceiling;
- `git rev-parse HEAD` changes after execution starts;
- CUDA/BF16 preflight fails on the white-box machine;
- a local model revision cannot be resolved exactly;
- a run-log audit fails;
- a real-world dataset release/hash/license lock is incomplete;
- generated paper artifacts cannot be traced to an audited run/analysis manifest.

A stopped, explicitly incomplete experiment is scientifically preferable to an untracked substitution or manually repaired paper result.
