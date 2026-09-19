# FairEval ECIR 2027 — one-week execution plan

Target effort: 5–8 focused hours/day. The objective is not to add arbitrary experiments; it is to execute the already frozen design, preserve provenance, and populate the paper only from audited artifacts.

The authoritative command-level contract is `docs/EXECUTION_RUNBOOK_2026-09-17.md`.

## Non-negotiable rules

Do not hand-edit frozen run plans or empirical paper numbers. Persistent invalidity is an outcome, not something to regenerate away. Do not change prompts, models, datasets, estimands, or statistical rules after seeing results without creating a new study version. FairSynth stays separate from human demographic/psychometric claims. Hosted and local evidence stay separate strata. Both real runners require the pre-execution seal.

## Day 1 — zero-call seal and data inventory

Pull a clean `ecir-2027-redesign` checkout. Install the analysis/dev/provider environment and run the repository preflights. Build:

```powershell
python scripts\build_preexecution_seal.py `
  --output-dir results\preexecution\seal-v1
```

The seal must prove zero hosted generation calls, no local model loading, a clean scientific worktree, exact scientific source hashes, a 12,960-cell hosted FairSynth plan, and a 12,960-cell local white-box plan.

In parallel, run `scripts/bootstrap_data_layout.py` and `scripts/check_raw_dataset_layout.py`. Acquire only exact releases named in `configs/dataset_acquisition.yaml`; leave unavailable releases explicitly blocked rather than substituting data.

## Day 2 — hosted gateway canary + GPU environment

Configure only the unified Zhizengzeng environment locally and run:

```powershell
python scripts\check_zhizengzeng_gateway.py --env-file .env --strict
```

If all six frozen IDs exist, dry-run and then execute one sealed cell per hosted family using `scripts/run_hosted_budgeted.py`. Keep the same output log and budget ledger. Hosted execution follows the 200 RMB normal stop, 250 RMB client-side emergency stop threshold, and 2 RMB reserve; never create a fresh ledger to evade the policy.

On AI Galaxy, prepare CUDA/PyTorch, install `requirements-local-gpu.txt`, and require:

```bash
python scripts/check_local_gpu.py --strict
```

No quantization is allowed in the canonical BF16 campaign merely to fit smaller hardware.

## Day 3 — full white-box FairSynth execution

Start with two sealed cells per local family. Audit the persisted rows, exact model revisions, seeds, candidate membership, and generation-score fields. If clean, run Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct separately in resume-safe batches until the full 12,960-cell plan is complete.

Use `scripts/run_whitebox_family.py` with the exact seal and Git SHA. Do not mix family state in one process and do not delete invalid rows.

## Day 4 — hosted FairSynth continuation + external release freezes

Continue the immutable hosted FairSynth plan while the budget policy permits. Budget truncation is missing planned coverage, not evidence for or against an RQ.

For each real dataset that has been acquired, review terms, generate a deterministic raw-release manifest with `scripts/hash_raw_release.py`, update `configs/dataset_releases.yaml` only from verified local evidence, build deterministic frozen instances, and verify them. Never invent a checksum.

## Day 5 — finalize FairSynth evidence and unlock real-world plans if possible

When both white-box family logs fully cover the core plan:

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

When hosted FairSynth execution is complete or policy-stopped, use `scripts/finalize_hosted_fairsynth.py`. These are the only normal paths from those run logs to paper numbers.

If all six real dataset locks are complete, require `scripts/check_pilot_readiness.py --strict` and create versioned real-world local/hosted plans. If not, keep the real-world tables pending and document the blockers.

## Day 6 — real-world RQ1–RQ4, only if data gates pass

Execute only immutable real-world plans whose exact data freezes exist. Run the registered RQ3 prompt/cue/order/cutoff/stochasticity plans without selecting favorable levels. Derive RQ4 identity-irrelevance prompting from the frozen core plan. Select contextual PAIR operating points on validation data only; test outcomes never choose or relax them.

Immediately audit and analyze each completed stratum. Keep hosted and local coverage explicit if the API budget truncates one lane.

## Day 7 — artifact-only paper and reproducibility package

Run the complete test/config/paper/result-contract chain and rebuild the anonymous Overleaf ZIP. Inspect the LNCS compile and page budget. Generated tables/figures must be traceable to audited machine-readable artifacts, plan hashes, data freezes, and the sealed code/specification.

Archive the scientific seal, environment manifest, exact code SHA, dataset manifests/hashes, immutable run plans, raw JSONL logs, budget ledger, audit outputs, analysis artifacts, generated paper tables/figures, and anonymous Overleaf bundle.

The schedule is successful even if external licensing/access blocks some real-world strata, provided that incompleteness is explicit and no frozen design is silently weakened to hide it.
