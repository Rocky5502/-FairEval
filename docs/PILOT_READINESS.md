# FairEval ECIR 2027 — Pilot Readiness Gate

The repository can be CI-green while some experimental strata are not yet executable. FairEval now separates readiness into two scopes: the project-owned FairSynth hosted/local campaigns, which can be sealed and executed before third-party data arrive, and the six-real-dataset RQ1–RQ4 campaign, which remains blocked until every external release is locally frozen and license-reviewed.

## Current state

`configs/dataset_releases.yaml` is deliberately `pre_freeze` for the six real-world datasets. Pending checksums/license fields are blockers, not findings, and must never be filled from memory or guesswork.

The hosted provider layer is no longer a collection of independent provider keys/endpoints. All six black-box families are frozen to one Zhizengzeng OpenAI-compatible gateway in `configs/models.yaml`, with exact gateway model IDs. The only hosted secret is the local `ZZZ_API_KEY` plus the frozen base URL.

The two local open-weight models and exact Hugging Face revisions are already frozen in `configs/local_models.yaml`.

## Zero-call scientific seal

Before any paid generation or local model weight loading, create:

```bash
python scripts/build_preexecution_seal.py \
  --output-dir results/preexecution/seal-v1
```

The seal requires a clean scientific worktree and records exact source hashes, Git SHA, canonical FairSynth freeze, 12,960-cell hosted plan, 12,960-cell white-box plan, pre-result paper bundle, and unresolved real-dataset blockers. Both real launchers verify the seal again before execution.

## FairSynth hosted readiness

The project-owned hosted FairSynth sanity layer does not require the six third-party releases. Before spending any hosted budget, set locally:

```dotenv
FAIREVAL_HOSTED_GATEWAY=zhizengzeng
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
ZZZ_API_KEY=<secret>
```

Then run:

```bash
python scripts/check_zhizengzeng_gateway.py --env-file .env --strict
```

All six exact frozen gateway IDs must be returned. No nearby model substitution is allowed.

The hosted budget policy is outcome-independent: 200 RMB normal stop, 250 RMB client-side emergency stop threshold, 2 RMB pre-cell reserve, and balance reconciliation around each persisted cell. The 250 RMB threshold is not represented as a provider-side atomic spending cap.

## Real-world dataset promotion

For each of the six datasets, acquire the exact upstream release under its terms, place it in the configured `data/raw/<dataset_id>` directory, review the license/terms, and create a deterministic raw-release manifest:

```bash
python scripts/hash_raw_release.py \
  --raw-dir data/raw/<dataset_id> \
  --output-manifest artifacts/raw_release_manifests/<dataset_id>.json
```

Only then copy the reported `directory_sha256` into `configs/dataset_releases.yaml`, set `license_reviewed: true`, and promote `release_status: frozen`. Build and verify the deterministic FairEval instance freeze afterward. A changed external release requires a new experiment version.

Use:

```bash
python scripts/check_raw_dataset_layout.py
python scripts/check_pilot_readiness.py --strict
```

for the full six-real-dataset readiness gate. `--strict` is intentionally expected to remain non-zero until those external locks and the local gateway environment are complete.

## White-box readiness

Before local GPU execution:

```bash
python scripts/check_local_gpu.py --strict
```

Use the exact sealed checkout and `scripts/run_whitebox_family.py`. The runner verifies the scientific seal, Git SHA, source hashes, plan hash, model family, and immutable cell IDs. The canonical FairSynth campaign is 12,960 generations and runs one frozen family per process.

## Pilot policy

Any variance/invalidity pilot may estimate operational quantities such as variance, persistent invalid-output rate, runtime, and cost. It must not use observed treatment-effect means to tune prompts, counterfactual definitions, model selection, statistical thresholds, or choose favorable conditions. Any scientific change after results requires an explicitly new study/plan version rather than mutation of the sealed run.

## Stop conditions

Stop rather than improvise if a real-dataset checksum differs from its lock, the live gateway omits an exact frozen model ID, source/seal verification fails, Git SHA changes after execution begins, CUDA/BF16 or exact local revision validation fails, the budget policy stops the hosted run, a run-log audit fails, or a result artifact cannot be traced to its sealed plan and code revision.

A versioned incomplete run is scientifically preferable to a silent model/data substitution or manually repaired result.
