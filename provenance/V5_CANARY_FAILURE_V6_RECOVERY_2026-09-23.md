# FairEval V5 Canary Failure and V6 Interface Recovery — 2026-09-23

## Status

The sealed V5 protocol-recovery canary completed all 144 planned cells (72 per local
model) under commit `b768a183a646a10909b240943751c9555f73d9da`, but the predeclared
promotion gate failed. V5 MUST NOT be scaled to the full campaign.

## Frozen V5 gate outcome

- Qwen2.5-7B-Instruct: 13/72 semantically valid exact-k candidate-valid rankings
  (18.0556%); 69/72 were strict JSON, showing that serialization compliance was
  not the main failure.
- Phi-3.5-mini-instruct: 63/72 semantically valid rankings (87.5%).
- Hard integrity checks passed: 144/144 rows present, no duplicate planned cell
  IDs, no candidate-ID mutation, no parser ambiguity, no commit mismatch, no
  model-revision mismatch, and no hardware mismatch.
- Main Qwen semantic errors: out-of-candidate item (53 rows), duplicate item IDs
  (7), and non-list `ranked_item_ids` (3).
- Main Phi semantic error: out-of-candidate item (9 rows).
- V5 promotion_allowed=false.

These values are protocol-diagnostic canary outcomes, not fairness-effect estimates.

## V6 correction

Inspection of the prompt interface showed that both `preference_history` and
`candidate_items` exposed item identifiers. That creates an avoidable interface
ambiguity: a model can emit a syntactically well-formed identifier copied from
history that is not eligible for ranking.

V6 therefore changes only the output interface:

1. history identifiers are omitted because they are not recommendation evidence;
2. candidates retain their frozen item identifiers;
3. an explicit `eligible_candidate_ids` whitelist is included in the prompt;
4. the output instruction requires exactly k unique IDs from that whitelist;
5. validation remains unchanged and strict: no ID rewriting, truncation,
   replacement, generative repair, or silent salvage;
6. one model generation remains permitted per cell.

The fairness manipulation, user evidence, candidate content/order, k, model
revisions, generation settings, and >=95% per-model promotion threshold are not
relaxed.

## Anti-overfitting safeguard

V6 uses a fresh deterministic 12-user FairSynth canary that is disjoint from the
12 users observed in V5. The V6 promotion gate is sealed before any V6 generation.
V5 outcomes are explicitly disclosed as known when V6 is defined.

If either model fails the same >=95% semantic exact-k candidate-validity threshold
on all 72 V6 cells, V6 must not scale.
