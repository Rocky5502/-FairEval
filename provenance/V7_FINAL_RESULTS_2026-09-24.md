# FairEval ECIR 2027 — V7 final-results provenance (2026-09-24)

## Completed local experiment

The submission-time local FairSynth experiment is the sealed V7 lean matrix:

- 120 canary-excluded FairSynth-360 users;
- 6 registered conditions;
- 2 seeded repetitions;
- 2 frozen local open-weight models;
- 2,880 planned and observed cells.

The final audit reports PASS and permits analysis. It records zero missing,
duplicate, or unknown planned-cell IDs; zero parser ambiguity; zero candidate-ID
mutation; zero deterministic selection-handle mapping mismatch; and zero commit,
model-revision, or hardware-provenance mismatch.

### Frozen identifiers

- execution commit: `83d7ac443a5797f7e0392134fa2ea6d62c7177cf`
- plan SHA-256: `9108f00789ffa9d20fab3ca97315cee8db68086eea501dabec32f18700ae1519`
- run-plan file SHA-256: `caf45d0ba753ea8d59987aaf035bf72e1df2c71735941025d2e50f70a6b95718`
- pre-execution seal SHA-256: `dd04810ebd0c40daf58bf68533e60a9acfb963ade012743580611ed13c54489e`
- merged run SHA-256: `9c91a3f936ce734b6460ad3030ccd12f261f6f655c0f4dfa168e0b3248780977`
- FairSynth inference SHA-256: `5d84314f9cb1ab8b92c514545b35cb5626b530931c87fc0cce1c7d770797df7d`

### Output reliability

- Phi-3.5-mini: 1,440/1,440 semantically valid; 1/1,440 strict-format
  valid; 1,439 rows deterministically envelope-normalized; no persistent
  semantic failures.
- Qwen2.5-7B: 1,371/1,440 semantically valid (95.2083%); 1,431/1,440
  strict-format valid; 69 persistent semantic failures:
  54 duplicate selections, 6 wrong-K outputs, 5 invalid-JSON responses,
  and 4 non-list payloads.
- No generative repair was used. Persistent invalid outputs remain system
  behavior, receive zero primary end-to-end utility, and contribute to invalid
  output disparity.

## Controlled FairSynth inference

Each model contributes 120 paired users per synthetic contrast.

### Meaningless synthetic identity

nDCG@10:

- Phi-3.5-mini: +0.00947, 95% CI [-0.00826, 0.02946], Holm p=0.6999.
- Qwen2.5-7B: -0.00809, 95% CI [-0.02654, 0.00922], Holm p=0.6999.

Recall@10:

- Phi-3.5-mini: +0.01167, 95% CI [-0.00542, 0.03000], Holm p=0.4351.
- Qwen2.5-7B: -0.00500, 95% CI [-0.03000, 0.01917], Holm p=0.7143.

Interpretation: no reliable nonzero utility effect from a synthetic identity
label that is relevance-invariant by construction.

### True versus shuffled synthetic OCEAN

nDCG@10:

- Phi-3.5-mini: -0.01372, 95% CI [-0.02899, 0.00023], Holm p=0.1445.
- Qwen2.5-7B: +0.00877, 95% CI [-0.01499, 0.03242], Holm p=0.4798.

Recall@10:

- Phi-3.5-mini: -0.01917, 95% CI [-0.04000, 0.00000], Holm p=0.1438.
- Qwen2.5-7B: +0.00333, 95% CI [-0.02667, 0.03333], Holm p=0.8754.

Interpretation: the controlled local experiment does not support a reliable
utility gain from true synthetic OCEAN over a shuffled-profile control.

## Local white-box diagnostics

Internal generation-score diagnostics are auxiliary and uncalibrated.

- Phi-3.5-mini valid-run mean token log-probability: -0.06698;
  descriptive Spearman rho with nDCG@10: -0.15794.
- Qwen2.5-7B valid-run mean token log-probability: -0.03096;
  descriptive Spearman rho with nDCG@10: -0.03651.

These values are not treated as calibrated confidence or confirmatory
uncertainty evidence.

## Hosted six-family pilot

The frozen hosted FairSynth lean campaign planned 1,080 cells. An outcome-blind
client budget guard stopped it after 99 cells (9.1667% coverage).

- ledger-wide balance movement: 64.9362 RMB;
- latest remaining gateway balance: 40.5347 RMB;
- every hosted family touched 3 users;
- every family has 3 structurally complete identity users and 2 structurally
  complete personality users.

The hosted pilot is retained only as operational feasibility and cost-provenance
evidence. Its paired-user coverage is insufficient for the preregistered hosted
FairSynth inference, so no hosted fairness/utility/personality effect is reported.

Hosted pilot plan SHA-256:
`9304f13b41c5caefc5f366fbe8a5646bb3170b3d46650f70779d424246132d14`.

## Paper policy

The final manuscript may report the completed V7 local FairSynth results and
the hosted operational pilot. It must not:

- pool FairSynth with real observed-demographic evidence;
- describe synthetic OCEAN as measured human psychometrics;
- infer hosted fairness effects from the 99-cell pilot;
- repair or drop the 69 Qwen semantic failures;
- describe local generation scores as calibrated uncertainty;
- fill missing real-world RQ1/RQ2/RQ4 results with pilot or synthetic values.
