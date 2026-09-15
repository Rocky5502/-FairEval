# FairEval ECIR 2027 Reproducibility Protocol

This document is the operational contract for the ECIR 2027 FairEval redesign. It is intentionally stricter than a normal README: once the pilot is frozen, changes to datasets, model IDs, provider endpoints, prompts, candidate construction, or analysis policy create a new experiment version.

## 1. Environment

Supported Python: **3.10+**.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
```

`requirements.txt` is the reviewer-facing installation manifest. `pyproject.toml` remains the package metadata source of truth. Provider SDKs are pinned because request schemas are part of the experimental apparatus.

Before a frozen pilot or confirmatory run, archive:

```bash
python --version
python -m pip freeze > artifacts/environment/pip-freeze.txt
git rev-parse HEAD > artifacts/environment/code-commit.txt
```

Do not commit API keys.

## 2. Credentials and endpoints

Copy `.env.example` to a local `.env` or export the variables in the shell. FairEval does not load or commit secrets automatically.

Required by family:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY` and `QWEN_BASE_URL`
- `LLAMA_PROVIDER_API_KEY`, `LLAMA_BASE_URL`, and `LLAMA_PROVIDER_NAME`

The exact Qwen region/endpoint and the Llama serving provider **must be frozen before the pilot**. A moving Llama provider alias is not acceptable for the final comparison.

## 3. Scientific configuration hierarchy

The principal pre-run configuration files are:

- `configs/study_design.yaml` — sampling, repetition, robustness, and inference policy.
- `configs/models.yaml` — six-family model panel, deliberation policy, and sampling policy.
- `configs/counterfactuals.yaml` — confirmatory and robustness demographic interventions.

Treat these files as pre-registration artifacts. After inspecting outcome means, do not edit them to improve a result.

## 4. Dataset freeze

Prepare each dataset independently. Example:

```bash
faireval prepare \
  --dataset movielens_1m \
  --raw-dir data/raw/movielens_1m \
  --output-dir data/frozen/movielens_1m \
  --users 20 \
  --candidate-set-size 50 \
  --max-history-items 20 \
  --seed 1729
```

Then verify the cryptographic manifest:

```bash
faireval verify-freeze --output-dir data/frozen/movielens_1m
```

Repeat for all six dataset adapters. Raw licensed datasets should not be committed unless their licenses permit redistribution.

## 5. Pilot rule

The pilot may estimate:

- metric variance;
- invalid-output rate;
- runtime;
- API cost;
- implementation failures.

The pilot must **not** use observed treatment-effect means to pick the confirmatory sample size, preferred prompt, preferred model, or preferred counterfactual. Final sample sizes and all confirmatory choices are frozen before confirmatory execution.

## 6. Immutable run plan

After all six dataset freezes are present:

```bash
faireval plan-core \
  --freeze-root data/frozen \
  --output-dir artifacts/plans/core-v1 \
  --counterfactuals configs/counterfactuals.yaml \
  --models configs/models.yaml \
  --seed 1729
```

This creates:

- `run_plan.jsonl` — immutable API cells;
- `plan_manifest.json` — dataset/config hashes and plan hashes.

Never hand-edit either file.

## 7. Dry run before spending API budget

```bash
faireval execute-plan \
  --plan-dir artifacts/plans/core-v1 \
  --freeze-root data/frozen \
  --output-jsonl artifacts/runs/core-v1.jsonl
```

Dry-run is the default and makes **zero provider calls**. Review the selected cell count, model families, and plan hash before adding `--execute`.

## 8. Provider execution

Record the exact code commit:

```bash
COMMIT=$(git rev-parse HEAD)
faireval execute-plan \
  --plan-dir artifacts/plans/core-v1 \
  --freeze-root data/frozen \
  --output-jsonl artifacts/runs/core-v1.jsonl \
  --code-commit-sha "$COMMIT" \
  --execute
```

For staged execution, use repeated `--family` arguments and/or `--max-cells`. Resume semantics are based on immutable `planned_cell_id` values; invalid model output still counts as an observed experimental outcome and is not silently regenerated until valid.

## 9. Provider policy

Core ranking uses the lowest practical deliberation configuration rather than pretending all providers expose identical decoding mechanics:

- OpenAI GPT-5.6 Terra: reasoning `none`.
- Claude Sonnet 5: thinking `disabled`.
- Gemini 3.8 Flash: thinking `low` (lowest supported setting in the frozen design).
- DeepSeek V4.1 Flash: thinking `disabled`.
- Qwen3.8-Max-0902: thinking `disabled`.
- Llama 4 Maverick: no extra hosted reasoning wrapper.

Requested main sampling profile is temperature `0.2`, top-p `1.0` where supported. Claude Sonnet 5 and Gemini 3.8 Flash use provider-default sampling where the current API does not accept the cross-provider controls. FairEval logs requested versus applied controls and does **not** claim decoding equivalence.

## 10. Prompt controls

RQ1–RQ3 use neutral audit prompts. Fairness coaching belongs only to explicitly named RQ4 mitigation modes.

Frozen robustness factors include:

- prompt template (`field_v2_a`, `field_v2_b`, `field_v2_c`);
- demographic cue representation;
- deterministic candidate-order permutations;
- repeated generations;
- K sensitivity.

No prompt should be selected after seeing which one yields the preferred conclusion.

## 11. Output integrity

Every persisted run row stores hashes and provenance for the prompt/request/response plus validity state. One format-only repair is permitted. The repair prompt may fix serialization but may not change the semantic ranking intentionally.

Malformed outputs are retained and contribute to invalid-output disparity analyses. Complete-case-only filtering is not the primary analysis.

## 12. Fairness and utility analysis guardrails

A changed list is a sensitivity observation, not automatically unfairness. Primary interpretation separates:

- held-out recommendation utility;
- counterfactual utility gap (CUG);
- counterfactual exposure gap (CEG) only where top-K item-group metadata is complete;
- group utility disparity (GUD);
- invalid-output disparity (IOD);
- personality value added (PVA) using true versus shuffled measured personality.

RBO/Jaccard are secondary ranking-sensitivity diagnostics.

## 13. Results integrity

Do not type experimental numbers manually into the paper. Result tables and figures remain `TBD` until generated from the frozen artifacts. Archive the script, input hashes, output figure/table file, and code commit used for every camera-ready numerical claim.

## 14. Release checklist

Before tagging a reproducible release:

1. `pytest -q` passes.
2. dataset freezes verify successfully;
3. model/provider endpoints are frozen;
4. `run_plan.jsonl` hash verifies;
5. `pip-freeze.txt` and code commit are archived;
6. no API keys or raw restricted data are present;
7. every result in the paper is traceable to an analysis artifact;
8. all comparison-table citations are verified against primary bibliographic sources.
