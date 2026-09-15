# FairSynth-360

**Status:** project-generated auxiliary benchmark for FairEval ECIR 2027.

FairSynth-360 contains **360 deterministic synthetic recommendation users/tasks** and a **120-item synthetic catalog**. It uses no external user data, names, text corpora, or personally identifiable information.

## Why it exists

The six real-world FairEval datasets cannot tell us with certainty that a demographic attribute is causally irrelevant to every user's preferences. FairSynth-360 supplies a controlled sanity check where that statement is true **by construction**.

- `synthetic_identity_group` is balanced across `A`, `B`, and `C` (120 users each).
- The identity assignment is generated independently from the user's latent preference vector and item relevance.
- Synthetic OCEAN values are partially informative about the latent preference vector.
- Relevance is generated only from the latent preference vector and item attributes.

Therefore:

1. identity-conditioned ranking differences have no ground-truth relevance justification in this benchmark;
2. true synthetic personality can contain useful preference information;
3. shuffled personality provides a controlled negative condition;
4. the benchmark can validate FairEval's sensitivity/consequence machinery without replacing real-world evaluation.

## Important scope language

The OCEAN vectors are **synthetic controls**, not measured human psychometrics. The A/B/C groups are **semantically meaningless synthetic labels**, not demographic categories. FairSynth-360 must never be pooled with MovieLens/Last.fm observed-demographic estimates or with the three measured-personality datasets when making real-world fairness or psychometric claims.

## Deterministic generation

Canonical dataset seed: `1729`

- users: `360`
- synthetic catalog: `120` items
- history pool: first `60` generated items
- candidate pool: remaining `60` generated items
- default materialized history: `8` items
- default candidate set: `30` items
- default relevant items: top `5` candidate-pool items by latent utility

The full generation specification is executable in `src/faireval/datasets/fairsynth360.py`. Materialize and cryptographically freeze it with:

```bash
python scripts/build_fairsynth360.py
python -m faireval.cli verify-freeze --output-dir data/frozen/fairsynth360
```

## Release policy

The dataset contains only project-generated synthetic numeric/tabular content. A public redistribution license should be explicitly selected and frozen before publication. Until then, the repository must not imply a license that the authors have not chosen.
