# FairEval Local Open-Weight / White-Box Track

This track extends the six hosted/API families with two locally executed open-weight models:

- `Qwen/Qwen2.5-7B-Instruct` — Apache-2.0
- `microsoft/Phi-3.5-mini-instruct` — MIT

It is a **separate transparency/reproducibility stratum**, not a size-matched causal comparison against the hosted models.

## Hardware target

The preferred single-GPU workstation target is an **NVIDIA GeForce RTX 5090 with 32 GB VRAM**. The experiment loads only one checkpoint at a time. Do not describe the target as an “RTX 5090 Ti” unless NVIDIA publishes such a product and the actual machine uses it.

Larger alternatives such as RTX 6000 Ada, L40S, A100, or H100 are acceptable, but the actual GPU/VRAM/CUDA/driver/PyTorch metadata must be logged. Changing hardware does not authorize changing the frozen model revision, prompt, candidate set, or analysis policy.

## Environment

Install the CUDA-matched PyTorch build for the workstation first, then:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# WSL/Linux: source .venv/bin/activate
pip install -r requirements-local-gpu.txt
pip install -e .
```

Run the zero-cost checks:

```bash
python scripts/check_config_consistency.py
python scripts/check_local_gpu.py
pytest -q
```

## Freeze exact model revisions

Before the first real local pilot, resolve and record the exact Hugging Face commit/revision for both repositories. Put those immutable identifiers in the local environment:

```text
QWEN25_LOCAL_REVISION=<exact commit>
PHI35_LOCAL_REVISION=<exact commit>
FAIREVAL_LOCAL_DTYPE=bfloat16
```

The pre-run manifest intentionally says `pin_exact_huggingface_commit_before_pilot` until this step is complete. Do not execute the final local study from a moving repository head.

## Build FairSynth-360

FairSynth-360 uses no external raw files:

```bash
python scripts/build_fairsynth360.py \
  --output-dir data/frozen/fairsynth360 \
  --users 360 \
  --candidate-set-size 30 \
  --max-history-items 8 \
  --seed 1729
```

Then verify the cryptographic freeze:

```bash
python -m faireval.cli verify-freeze \
  --output-dir data/frozen/fairsynth360
```

## Compile the local immutable plan

After the six real-world freezes also exist under `data/frozen/`:

```bash
python scripts/plan_local_open_weight.py \
  --freeze-root data/frozen \
  --output-dir results/plans/local-open-weight-v1 \
  --fairsynth-users 360 \
  --repetitions 3
```

For a FairSynth-only first smoke run, add `--no-real-world`.

Every local plan cell contains a deterministic generation seed. The seed is part of the cell hash and is rechecked before white-box analysis.

## Zero-call inspection

The shared executor remains dry-run by default:

```bash
python -m faireval.cli execute-plan \
  --plan-dir results/plans/local-open-weight-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/raw/local-open-weight-v1.jsonl \
  --family qwen25_local \
  --max-cells 10
```

Only after reviewing the summary should `--execute` be added with the exact code commit SHA.

## Execute one model at a time

Example:

```bash
python -m faireval.cli execute-plan \
  --plan-dir results/plans/local-open-weight-v1 \
  --freeze-root data/frozen \
  --output-jsonl results/raw/local-open-weight-v1.jsonl \
  --family qwen25_local \
  --code-commit-sha <git-commit> \
  --execute
```

After completion, release the model/GPU memory before starting `phi35_local`. Do not run both checkpoints concurrently in the canonical 32-GB profile.

## Analyze FairSynth separately

```bash
python scripts/analyze_fairsynth360.py \
  --output-jsonl results/raw/local-open-weight-v1.jsonl \
  --plan-dir results/plans/local-open-weight-v1 \
  --freeze-root data/frozen \
  --output-dir results/analysis/fairsynth360-v1
```

FairSynth identity A/B/C is semantically meaningless and independent of relevance by construction. Synthetic OCEAN is **not measured human personality**. These results are never pooled with real-world RQ1/RQ2 inference.

## Analyze local internal scores

```bash
python scripts/analyze_local_whitebox.py \
  --output-jsonl results/raw/local-open-weight-v1.jsonl \
  --plan-dir results/plans/local-open-weight-v1 \
  --freeze-root data/frozen \
  --output-dir results/analysis/local-whitebox-v1
```

The analyzer reports token log-probability, NLL/perplexity, and top-1/top-2 generation-score margin against utility/invalidity descriptively. These are **uncalibrated auxiliary generation-score diagnostics**, not a probability that a ranking is correct and not directly comparable with unavailable internals from closed APIs.
