# FairEval ECIR 2027 — Pilot Readiness Gate

The repository can be **CI-green** while still being **not ready for paid/API execution**. This is intentional. The pilot must not start until dataset releases, provider endpoints, credentials, model semantics, and frozen instances are all locked.

## Current state

`configs/dataset_releases.yaml` is deliberately `pre_freeze`. Its checksum and license-review fields are placeholders marked pending; they are not findings and must never be filled from memory or guesswork.

The Meta/Llama checkpoint identity is selected, but the exact serving provider/endpoint is still a pilot blocker until it is frozen in `configs/models.yaml` and the corresponding environment variables.

## Promotion sequence

For **each of the six datasets**:

1. Download the exact upstream release into the configured `data/raw/<dataset_id>` directory.
2. Review the upstream license/terms for that exact release and record the result in `configs/dataset_releases.yaml`.
3. Generate a canonical raw-release manifest:

   ```bash
   python scripts/hash_raw_release.py \
     --raw-dir data/raw/<dataset_id> \
     --output-manifest artifacts/raw_release_manifests/<dataset_id>.json
   ```

4. Copy the reported `directory_sha256` into the dataset's `raw_sha256` field.
5. Set `release_status: frozen` only after the release identity, local directory and license review are all verified.
6. Build the FairEval frozen instances with the dataset adapter.
7. Verify the generated freeze manifest and hashes before creating the run plan.

A dataset release change after this point requires a **new experiment version** and new frozen instances; it must never be swapped silently after seeing outcomes.

## Provider freeze

Before six-family pilot execution:

- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, and `LLAMA_PROVIDER_API_KEY` must be present locally.
- `QWEN_BASE_URL` must identify the exact Model Studio regional endpoint used for the study.
- `LLAMA_BASE_URL` and `LLAMA_PROVIDER_NAME` must identify the selected Llama serving provider.
- The Llama host's exact output-token request field must be re-verified against `configs/models.yaml`.
- No secrets are committed to GitHub. `.env.example` contains names only.

## Readiness commands

Repository/schema health (safe for CI and does **not** claim pilot readiness):

```bash
python scripts/check_config_consistency.py
python scripts/check_pilot_readiness.py
pytest -q
```

Strict local gate immediately before spending API budget:

```bash
python scripts/check_environment.py
python scripts/check_pilot_readiness.py --strict
```

`--strict` must exit successfully. Do not bypass it simply to make the run start.

## Run-plan gate

After all six dataset freezes are complete:

```bash
faireval plan-core \
  --freeze-root artifacts/frozen \
  --counterfactuals configs/counterfactuals.yaml \
  --models configs/models.yaml \
  --output-dir artifacts/plan \
  --seed 2027
```

Then inspect the immutable plan and execute a **zero-call dry run** first. Only after plan counts, family IDs, frozen hashes, prompt settings, reasoning/thinking settings, sampling policies, and output-token fields agree should `--execute` be used.

## Pilot policy

The variance/invalidity pilot uses **20 users per dataset**. It may estimate:

- variance;
- invalid-output rate;
- runtime;
- token/API cost.

It must **not** use the observed treatment-effect mean to tune confirmatory sample size, prompts, counterfactual definitions, model selection, or significance thresholds. Confirmatory N and split hashes are frozen after the pilot and before confirmatory outcomes are observed.

## Stop conditions

Stop rather than improvising if any of the following occurs:

- a dataset checksum/release identity differs from the lock;
- a provider silently resolves a different model/backend;
- reasoning/thinking mode cannot be applied as frozen;
- the runtime token-limit field differs from the immutable plan;
- a dataset adapter produces a different freeze hash with unchanged inputs/code;
- the output log fails `scripts/audit_run_log.py`;
- a result file mixes code commits or run plans.

These are reproducibility failures, not inconveniences to work around.
