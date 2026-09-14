# FairEval — ECIR 2027 Reboot

**Working submission title:** *FairEval: Separating Personalization from Discrimination in Personality-Aware LLM Recommendations*

This repository is the clean-room research reboot of FairEval for **ECIR 2027**. The study is intentionally redesigned rather than treated as a cosmetic extension of the earlier paper.

## Core scientific change

The old setup could interpret any recommendation-list change after adding identity/personality information as unfairness. The ECIR study instead separates:

1. **Behavioral shift** — did the ranked list change?
2. **Preference alignment / utility** — did the change improve or harm held-out user preference utility?
3. **Counterfactual identity effect** — does changing identity while holding preferences and candidates fixed change utility/exposure?
4. **Grounded personality value** — does a *measured* Big Five profile help beyond a neutral prompt and a shuffled-personality negative control?

A changed list is therefore **not automatically labeled unfair**. A change may be legitimate personalization when it improves preference alignment; it becomes concerning when identity causes unjustified quality/exposure gaps, stereotype-linked shifts, or other counterfactual harms.

## Four research questions

- **RQ1 — Personalization vs. discrimination.** When demographic or personality context changes an LLM ranking, when is the change preference-aligned personalization, and when does it create an unjustified counterfactual utility/exposure gap while user history and candidate items are fixed?
- **RQ2 — Grounded personality.** Do measured Big Five traits provide recommendation value beyond preference history alone, and are any gains robust to shuffled-personality and one-trait counterfactual controls across personality strata?
- **RQ3 — Generalization and reliability.** How stable are utility, behavioral-shift, and fairness conclusions across six datasets, six LLM families, prompt templates, cutoffs, and repeated generations?
- **RQ4 — Mitigation.** Can lightweight identity-aware prompting and preference-aligned counterfactual re-ranking reduce harmful gaps while preserving recommendation utility and legitimate personality-driven gains?

## Planned benchmark

### Datasets / domains

The benchmark uses two complementary tracks.

**Personality-grounded track**
- GroupLens Personality 2018 — movies + measured Big Five
- Music Master / Beyond the Big Five — music + BFI/BFI-2
- REASONER — short video + CBF-PI-15 Big Five responses

**Generalization / counterfactual track**
- MovieLens-1M — movies + interaction history / demographic fields
- Last.fm-1K or HetRec Last.fm — music + interaction history; profile fields where licensing/schema permit
- MIND — news + real click histories and impression candidate sets

Every dataset adapter must preserve its license and document exactly which attributes are observed versus synthetically counterfactualized.

### LLM families

One pinned model snapshot per vendor family, selected before the main run:
- OpenAI
- Anthropic Claude
- Google Gemini
- DeepSeek
- Alibaba Qwen
- Meta Llama

Exact model IDs, API dates, parameters, and provider revisions will be frozen in a run manifest. Provider adapters must not silently substitute newer aliases.

## Evaluation principles

- Candidate-constrained ranking: models choose only from auditable candidate item IDs.
- Real preference histories and held-out relevance labels wherever the dataset permits.
- Big Five/OCEAN numeric profiles, not ad-hoc adjectives such as “introverted engineer.”
- Syntax-controlled prompt templates with explicit neutral/null fields.
- Counterfactual pairs hold history, candidate set, task, and syntax constant.
- Multiple generations; no single-pass conclusions.
- Invalid/malformed outputs are logged and reported by model/condition/group, not silently discarded.
- Utility and fairness are reported jointly; list similarity is a diagnostic, not a fairness definition.
- Paired uncertainty intervals, multiple-comparison correction, effect sizes, and mixed-effects/variance analysis are planned before looking at headline results.

## Repository layout

```text
configs/          experiment/model manifests
src/faireval/     benchmark implementation
tests/            unit tests for metrics and prompt invariants
docs/             ECIR research blueprint and reviewer-to-design traceability
paper/            LNCS manuscript scaffold; numerical results remain TBD until generated
results/          generated outputs only (not committed if large/sensitive)
```

## Reproducibility rule

No result is hand-entered into the paper. Final tables and figures must be generated from immutable result manifests containing dataset split hashes, prompt/template versions, model IDs, generation settings, random seeds (where supported), timestamps, parser failures, and code commit SHA.

## Status

Research-design reboot started September 2026. Current priority: freeze the ECIR design, implement the benchmark skeleton, run pilot validation, then execute the preregistered matrix before the ECIR full-paper deadline.
