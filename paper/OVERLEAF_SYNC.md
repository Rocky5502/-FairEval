# Overleaf synchronization contract — final-results state

The repository branch `ecir-2027-final-results` is the paper-facing source of truth.
The frozen V7 experiment itself remains tied to execution commit
`83d7ac443a5797f7e0392134fa2ea6d62c7177cf`; paper polishing must never mutate or
reinterpret those run logs.

## Final empirical evidence included in Overleaf

The final manuscript consumes only audited generated artifacts when they exist.
Missing empirical strata are omitted rather than replaced by TBD or hand-entered values.

Included completed evidence:

- complete V7 local FairSynth matrix: 2,880/2,880 cells;
- 120 canary-excluded FairSynth users, 6 conditions, 2 seeded repetitions, 2 local models;
- audited local FairSynth paired inference;
- audited local white-box diagnostic summary;
- audited V7 output-reliability summary;
- outcome-blind hosted six-family operational pilot: 99/1,080 planned cells.

The hosted pilot is operational feasibility and cost provenance only. It is not used
for fairness, utility, ranking, or personality-effect inference because paired-user
coverage is below the preregistered hosted inference target.

## Generated tables

The final result entrypoint is `results_contract_table.tex`. It conditionally includes:

- `generated/rq12_inference_table.tex` only if a qualified real-world artifact exists;
- `generated/rq34_results_table.tex` only if qualified RQ3/RQ4 artifacts exist;
- `generated/fairsynth_hosted_table.tex` only if complete hosted FairSynth inference exists;
- `generated/fairsynth_local_table.tex` from the audited V7 local inference;
- `generated/v7_execution_quality_table.tex` from the audited V7 main-run report;
- `generated/whitebox_summary_table.tex` from the audited V7 white-box summary.

The legacy files under `paper/result_tables/` remain bundled as study contracts but are
not auto-included in the final manuscript when the corresponding empirical artifact is absent.

## V7 local result provenance

Frozen identifiers:

- V7 execution commit: `83d7ac443a5797f7e0392134fa2ea6d62c7177cf`
- plan SHA-256: `9108f00789ffa9d20fab3ca97315cee8db68086eea501dabec32f18700ae1519`
- run-plan file SHA-256: `caf45d0ba753ea8d59987aaf035bf72e1df2c71735941025d2e50f70a6b95718`
- pre-execution seal SHA-256: `dd04810ebd0c40daf58bf68533e60a9acfb963ade012743580611ed13c54489e`
- merged run SHA-256: `9c91a3f936ce734b6460ad3030ccd12f261f6f655c0f4dfa168e0b3248780977`
- FairSynth inference SHA-256: `5d84314f9cb1ab8b92c514545b35cb5626b530931c87fc0cce1c7d770797df7d`

Full interpretation and paper-use restrictions are recorded in
`provenance/V7_FINAL_RESULTS_2026-09-24.md`.

## Hosted pilot provenance

The six-family hosted FairSynth lean campaign planned 1,080 cells and stopped
outcome-blind after 99 cells under the frozen client budget guard. The authoritative
summary is `provenance/hosted_fairsynth_pilot99_operational_summary.json`.

The exact current hosted model IDs come from `configs/models.yaml`, including
`qwen3.8-max`; do not revive the rejected earlier Qwen identifier.

## Integrity rules

- keep anonymous author/institution metadata until camera-ready;
- never type empirical result values into manuscript prose without matching audited artifacts;
- never pool FairSynth with real observed-demographic evidence;
- never describe synthetic OCEAN as measured human psychometrics;
- never infer hosted fairness effects from the 99-cell operational pilot;
- never repair or drop persistent invalid outputs from the V7 main run;
- never describe local token-score diagnostics as calibrated uncertainty;
- never select RQ4 operating points from test outcomes;
- never describe the 250 RMB client-side threshold as a provider-enforced atomic cap.

## Final Overleaf build

Run the repository preflights/CI, then build:

```bash
python scripts/build_overleaf_bundle.py \
  --output dist/FairEval_ECIR2027_Overleaf.zip
```

The resulting ZIP must include the audited V7 generated tables and the hosted pilot
operational table/figure when present. The bundle remains double blind.


## Submission closeout

The paper-facing submission scope is frozen in
`configs/submission_scope_ecir2027.yaml`. Run:

```bash
python scripts/check_submission_closeout.py
```

before final upload. A passing closeout certifies the completed V7 local evidence,
the outcome-blind hosted operational pilot, paper/result guards, tests/preflights,
and regenerated anonymous Overleaf bundle. It does not fabricate or require the
deferred six-real-dataset release locks.
