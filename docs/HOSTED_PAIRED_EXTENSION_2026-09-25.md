# Hosted FairSynth paired-completion extension — 2026-09-25

## Purpose

The first hosted FairSynth execution stopped outcome-blind after 99/1,080 planned
cells because of the original cost guard. Only coverage and gateway-balance
movement were inspected. Recommendation rankings, utility values, paired effect
sizes, confidence intervals, and p-values from those 99 cells were not used to
design this extension.

The extension exists to replace scattered operational coverage with a small but
complete cross-family paired sample that can support synthetic FairSynth
identity/personality inference.

## Frozen target

The extension targets exactly **9 complete users per hosted family**, with exact
synthetic-identity balance:

- A: 3 users
- B: 3 users
- C: 3 users

For every selected user, all six FairSynth conditions must be complete for all
six hosted families. The resulting analysis geometry is therefore:

- 9 users
- 6 conditions
- 6 hosted model families
- 1 generation per cell
- 324 target cells total
- 54 user/model identity pairs
- 54 user/model personality pairs

## Selection rule

Selection is operational and outcome-blind.

Within each synthetic identity group:

1. reuse previously **fully complete** 36-cell users;
2. exclude every partially completed user;
3. fill the remaining slots with previously untouched users in immutable
   parent-plan order.

This rule maximizes reuse while preventing one matched user's six conditions
from spanning two execution commits/specifications.

The extension planner binds the prior run SHA-256, parent plan SHA-256,
operational-summary SHA-256, selected target users, pending-cell IDs, and
projected cost into a new plan manifest.

## Budget rule

The extension uses a new ledger rather than changing the historical pilot
ledger. The default extension policy is:

- normal extension stop target: 190 RMB
- emergency extension threshold: 200 RMB
- request reserve: 2 RMB

The planner estimates pending cost from the already observed per-family
request-window cost averages and requires the projection multiplied by a 1.20
safety factor to fit below the normal target.

These values are experiment-operation metadata. They are retained in the
artifact/provenance and are not required in the main ECIR narrative.

## Pre-execution seal

A new V4 pre-execution seal is mandatory. It explicitly records:

- the prior 99-cell hosted pilot is known operationally;
- prior hosted scientific outcomes were not inspected;
- extension selection used only coverage/cost;
- local V7 results are already known;
- no hosted extension result was observed before the seal;
- the exact extension plan and scientific-source commit are frozen.

## Analysis and paper claims

After the extension completes, the old and new raw logs are audited separately.
They are never concatenated and then treated as a single same-commit raw run.

For analysis only:

- fully complete target users from the original run are scored under the
  original parent plan;
- new target users are scored under the extension plan;
- scored rows are combined only after each raw log passes its own provenance
  audit;
- every selected user must have all six conditions for all six families.

Paper-facing hosted outputs are generated automatically:

- `paper/generated/fairsynth_hosted_table.tex`
- `paper/generated/fairsynth_hosted_summary.tex`
- `paper/figures/fairsynth_hosted_profiles.pdf`

The table reports paired nDCG@10 and Recall@10 identity/personality effects,
95% bootstrap confidence intervals, and Holm-adjusted paired sign-flip p-values.

### Allowed interpretation

The identity contrast is a **controlled fairness-sanity test** because synthetic
identity is deliberately irrelevant to relevance.

The personality contrast is a **synthetic personalization-value sanity test**.

### Not allowed

The hosted extension must not be described as:

- real-world demographic fairness evidence;
- measured-human-personality evidence;
- a ranking of which vendor/model is "fairest";
- proof that a model is generally fair or unfair;
- evidence from 30 users or the full 1,080-cell plan.

The correct scope is nine balanced FairSynth users per hosted family.
