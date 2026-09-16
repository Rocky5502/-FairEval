# FairEval ECIR 2027 - one-week execution plan

Target effort: **5-8 focused hours/day for seven days**.

The goal of this week is not to create arbitrary extra experiments. It is to move from a CI-green, pre-run repository to a frozen, auditable empirical package and then populate the paper only from generated artifacts.

## Non-negotiable rules

1. Do not hand-edit a frozen run plan or result number.
2. Do not rerun malformed outputs until they become valid; persistent invalidity is an outcome.
3. Do not use pilot treatment-effect means to chase significance.
4. Do not change prompts/models/datasets after seeing confirmatory results without versioning a new study.
5. FairSynth-360 remains a controlled synthetic sanity test and is never pooled with real demographic or measured-psychometric evidence.
6. Qwen2.5-7B and Phi-3.5-mini use the exact frozen Hugging Face commits in `configs/local_models.yaml`.
7. Keep the PR draft until experiment freezes and paper artifacts are complete.

---

## Day 1 - machine, repository, data-source locks

### 1. Clone/update and create the environment

Windows PowerShell:

```powershell
git clone https://github.com/Rocky5502/-FairEval.git FairEval
cd FairEval
git checkout ecir-2027-redesign
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For the local open-weight track, first install the CUDA-matched PyTorch build for the machine, then:

```powershell
python -m pip install -r requirements-week1.txt
```

### 2. Run the zero-provider-call bootstrap

```powershell
python scripts/week1_bootstrap.py
```

This performs no API/model-generation calls. Keep:

- `results/bootstrap/week1_status.json`
- `results/environment_manifest.json`
- `results/smoke/fairsynth_oracle_v1.json`

### 3. GPU check

```powershell
python scripts/check_local_gpu.py
```

If CUDA/PyTorch is not ready, fix that before downloading the local checkpoints. Do not introduce a quantized protocol merely to force a smaller GPU; quantization would be a separately versioned experiment.

### 4. Dataset acquisition

Create/check the canonical layout:

```powershell
python scripts/bootstrap_data_layout.py
python scripts/check_raw_dataset_layout.py
```

Download only the exact upstream releases named in `configs/dataset_acquisition.yaml`. Record license/terms review and use `scripts/hash_raw_release.py` before marking anything frozen.

**Day-1 exit criterion:** zero-call bootstrap green; exact local revisions frozen; raw dataset directories created; each external dataset either downloaded or has an explicit acquisition blocker.

---

## Day 2 - freeze datasets and compile pilot plans

For each real dataset, first validate the raw layout, then freeze **20 pilot users** using deterministic seeds. Example:

```powershell
faireval prepare --dataset movielens_1m --raw-dir data/raw/movielens_1m --output-dir data/frozen/movielens_1m --users 20 --candidate-set-size 50 --max-history-items 20 --seed 1729
faireval verify-freeze --output-dir data/frozen/movielens_1m
```

Repeat for all six real-world datasets.

Freeze FairSynth-360 separately using the project generator and its registered defaults/full-user scope.

Then run:

```powershell
python scripts/check_pilot_readiness.py --strict
```

Do not proceed to hosted calls until the strict gate is green except for the explicitly separate local/FairSynth track.

Compile plans and dry-run them before making calls:

```powershell
faireval plan-core --freeze-root data/frozen --output-dir results/plans/core-pilot-v1 --counterfactuals configs/counterfactuals.yaml --models configs/models.yaml --seed 1729
faireval execute-plan --plan-dir results/plans/core-pilot-v1 --freeze-root data/frozen --output-jsonl results/raw/core-pilot-v1.jsonl --max-cells 5
```

Compile the local open-weight plan and RQ3/RQ4 extra plans using the repository scripts after their required freezes exist.

**Day-2 exit criterion:** immutable pilot freezes and plans exist, hashes verified, every dry run succeeds, planned cell counts reviewed.

---

## Day 3 - local open-weight + FairSynth execution

Run **one local model at a time**.

Priority order:

1. Phi-3.5-mini on a small FairSynth smoke subset.
2. Qwen2.5-7B on the same subset.
3. Full FairSynth-360 local runs.
4. Selected real-dataset replication cells only after the local pipeline is proven stable.

Always capture the exact code SHA:

```powershell
$sha = git rev-parse HEAD
```

Use immutable plans and resume semantics. Never manually delete an invalid result row to obtain a cleaner run.

After the local run:

```powershell
python scripts/audit_run_log.py ...
python scripts/analyze_fairsynth360.py ...
python scripts/analyze_local_whitebox.py ...
```

Inspect white-box diagnostics as **exploratory, uncalibrated** quantities only.

**Day-3 exit criterion:** local model pipeline produces audited raw logs + FairSynth inference + white-box summaries, or a precisely documented hardware/software blocker.

---

## Day 4 - hosted 20-user pilot

Only after strict readiness passes, run a small number of cells per provider first, audit them, and then execute the pilot plan.

Pilot purposes only:

- invalid-output rate;
- runtime/latency;
- token/cost profile;
- within-user/model variability needed for final N;
- provider schema/format behavior.

Do **not** choose N from whichever treatment mean looks promising.

At the end of the pilot freeze:

- final confirmatory N (minimum 60, target 100, maximum 120 as registered);
- exact user/split hashes;
- exact candidate sets and order;
- exact endpoints/regions;
- exact code commit;
- exact prompt/cue/model generation semantics.

**Day-4 exit criterion:** pilot report + frozen confirmatory N/protocol. No confirmatory result interpretation yet.

---

## Day 5 - confirmatory RQ1/RQ2 + RQ3 robustness

Execute the frozen confirmatory plan with exact resume semantics.

Immediately after runs:

```powershell
python scripts/audit_run_log.py ...
python scripts/analyze_core.py ...
python scripts/analyze_rq2_traits.py ...
python scripts/analyze_rq3.py ...
```

Primary outputs:

- scored run rows;
- user-condition aggregates;
- RQ1/RQ2 paired estimands;
- C5 one-trait effects;
- inference/Holm artifacts;
- RQ3 factor-level reliability artifacts.

Do not manually select a prompt/cue/order level because it looks better.

**Day-5 exit criterion:** RQ1-RQ3 frozen artifacts complete and audited.

---

## Day 6 - RQ4 mitigation + generated paper results

Build and execute the separate identity-irrelevance prompting plan, then analyze it.

Run contextual PAIR only with the preregistered grid. The one global operating point must be selected on deterministic 20% validation users, retain at least 95% of baseline validation nDCG, and then minimize absolute CUG. Test outcomes never choose or relax the point.

Generate all paper-facing evidence from artifacts:

```powershell
python scripts/render_result_tables.py ...
python scripts/build_result_figures.py ...
```

The two main result tables are mandatory anchors. The four compact supporting tables are promoted into the main 12-page paper only if they materially support the final story.

**Day-6 exit criterion:** all RQ1-RQ4 numerical tables/figures generated automatically; no hand-entered empirical number.

---

## Day 7 - paper, 12-page layout, reproducibility release

Run all checks again:

```powershell
pytest -q
python scripts/check_config_consistency.py
python scripts/check_paper_source.py
python scripts/check_result_table_contracts.py
python scripts/build_overleaf_bundle.py --output dist/FairEval_ECIR2027_Overleaf.zip
```

Compile the exact bundle with Springer LNCS/BibTeX and visually inspect every page.

Final paper-layout priorities:

1. keep the two big RQ result tables before References;
2. keep only result figures that add non-tabular evidence (likely RQ1 geometry and RQ4 Pareto first);
3. compress repetitive prose rather than shrinking fonts;
4. main text including figures/tables must fit the ECIR limit; references remain separate according to the venue rule;
5. retain the four compact generated tables even if only some are promoted to the main paper.

Build a final provenance package containing manifests, hashes, exact code SHA, generated result tables, result figures, environment manifest, and the anonymous Overleaf ZIP.

**Day-7 exit criterion:** clean compile, page budget satisfied, CI green, frozen artifacts archived, PR still unmerged until final human review.

---

## Recommended daily rhythm (5-8 hours)

- 45 min: reproduce yesterday's state + check logs/manifests.
- 3-5 h: one major experiment/execution block.
- 60-90 min: audit/analyze generated artifacts immediately.
- 45-60 min: paper/repo synchronization and commit.
- final 15 min: write exact next starting command so the following day begins without rediscovery.

The week is feasible if external dataset access/provider credentials are available promptly. The main schedule risk is not coding; it is dataset licensing/download friction, provider access, GPU/CUDA setup, and API throughput. Do not compromise the frozen design to compensate for those delays.
