# FairEval Local Open-Weight / White-Box Track

This is the required, separately reported white-box replication stratum for ECIR 2027. It extends the six hosted/API families with exactly two frozen open-weight models:

- `Qwen/Qwen2.5-7B-Instruct` — Apache-2.0;
- `microsoft/Phi-3.5-mini-instruct` — MIT.

The exact Hugging Face revisions are already frozen in `configs/local_models.yaml` and enforced by the local provider. Do not replace either revision, add another model, quantize the canonical run, or tune the panel after observing results. This is not a size-matched causal comparison with the hosted families.

## Hardware

The canonical protocol loads one model at a time in BF16. A 32–48+ GiB NVIDIA GPU is preferred; larger AI Galaxy instances such as A100/H100/L40S-class hardware are acceptable. Record the actual GPU, VRAM, CUDA, driver, PyTorch, Transformers, and code commit. Hardware changes do not authorize changes to prompts, candidates, model revisions, generation settings, or analysis.

## Zero-call seal before GPU execution

From a clean checkout, install the project and build the scientific seal before any model weights are loaded:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements-local-gpu.txt
pip install -e .

python scripts/build_preexecution_seal.py \
  --output-dir results/preexecution/seal-v1
```

The seal verifies a clean scientific worktree, hashes all tracked scientific source files, regenerates the canonical FairSynth freeze and both immutable plans, records the exact Git commit, and proves that no hosted generation or local model loading occurred while the seal was created. Real execution later rechecks the sealed source hashes, code commit, plan hash, and planned-cell count.

## Canonical FairSynth campaign

The sealed core plan contains exactly:

```text
360 FairSynth users
× 6 registered conditions
× 3 seeded repetitions
× 2 frozen local models
= 12,960 generations
```

The builder is zero-call:

```bash
python scripts/build_whitebox_campaign.py \
  --freeze-root data/frozen \
  --output-root results/plans/whitebox-full-v1
```

Each local cell carries a deterministic generation seed inside its immutable cell hash.

## GPU preflight and canary

```bash
nvidia-smi
python scripts/check_local_gpu.py --strict

SHA=$(git rev-parse HEAD)
python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family qwen25_local \
  --output-jsonl results/runs/whitebox-qwen25-v1.jsonl \
  --max-cells 2
```

The first command is a dry run. For the two-cell real canary, use the same command with:

```bash
  --preexecution-seal results/preexecution/seal-v1/PREEXECUTION_SEAL.json \
  --code-commit-sha "$SHA" \
  --execute
```

Repeat the canary for `phi35_local` using its own output JSONL. Only scale after the persisted rows pass the run-log/seed/provenance checks.

## Full resume-safe execution

Run one family per process, for example in 250–1000-cell operational batches. Batch size changes checkpoint frequency only; it does not change the immutable scientific plan.

```bash
SHA=$(git rev-parse HEAD)
python scripts/run_whitebox_family.py \
  --plan-dir results/plans/whitebox-full-v1/core \
  --freeze-root data/frozen \
  --family qwen25_local \
  --output-jsonl results/runs/whitebox-qwen25-v1.jsonl \
  --max-cells 500 \
  --preexecution-seal results/preexecution/seal-v1/PREEXECUTION_SEAL.json \
  --code-commit-sha "$SHA" \
  --execute
```

Use the analogous command for `phi35_local`. Completed `planned_cell_id` values are skipped exactly on resume. Persistent invalid outputs remain experimental outcomes and are not repeatedly regenerated until valid.

## Finalize directly into the paper

Only after both family logs fully cover the sealed core plan:

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

The finalizer audits coverage/provenance, merges rows in immutable plan order, analyzes the declared white-box diagnostics, renders the artifact-derived LaTeX table, and rebuilds the anonymous Overleaf bundle. Do not manually average or type numerical values into the paper.

## RQ3/RQ4 and real-world replication

Real-world execution stays blocked until all six exact third-party releases are locally acquired, license-reviewed, hashed, and frozen. Once those locks exist, version a new campaign with `--include-real-world` and compile the registered local RQ3 plan with `scripts/plan_whitebox_rq3.py`. RQ4 identity-irrelevance prompting is a derived immutable plan; contextual PAIR remains validation-selected post-processing.

White-box token log-probability, NLL/perplexity, and top1–top2 margin are uncalibrated auxiliary generation-score diagnostics. They are not probabilities of correctness and are never pooled with unavailable hosted internals.
