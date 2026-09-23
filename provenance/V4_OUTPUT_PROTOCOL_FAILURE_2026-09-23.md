# V4 Output-Protocol Failure and V5 Recovery

## Status

The frozen V4 white-box campaign is preserved unchanged at commit
`05d90102a3de2c90310eee76b75757d745d6c664`. Its run, seal, analysis, and
generated artifacts must not be overwritten by V5.

V4 completed its planned generation campaign with intact run provenance, but
post-run inspection exposed a systematic output-interface failure: the validator
required a direct top-level JSON object, while real model responses frequently
used Markdown fences or otherwise violated the exact output envelope. The V4
generative "format repair" path was additionally found capable of changing item
ID strings (including leading-zero semantics), which violates the intended
format-only contract.

## Scientific treatment

V4 is retained as:

- a completed provenance-clean generation campaign;
- evidence of a benchmark-interface/protocol failure;
- a source for explicitly post-hoc forensic sensitivity analysis.

V4 is **not** silently reparsed and relabeled as the preregistered primary
white-box result. The script `scripts/analyze_whitebox_v4_forensic.py` reads
only the original first responses, ignores generative repair outputs, performs
deterministic envelope parsing, and writes a separate post-hoc artifact with
source hashes.

## V5 recovery

V5 is a separately versioned protocol revision. It:

1. explicitly requests exactly K candidate IDs and exact ID copying;
2. separates strict serialization compliance from semantic ranking validity;
3. makes exactly one generation call per experimental cell;
4. removes LLM/generative repair from canonical execution;
5. allows only deterministic, envelope-only normalization;
6. rejects parser ambiguity;
7. treats any candidate-ID mutation as a hard failure.

Before any V5 scale-up, a sealed FairSynth-360 canary contains 12 users x 6
conditions x 1 repetition x 2 models = 144 generations. The promotion rule is
fixed before canary results: each model must reach at least 95% semantic exact-K,
candidate-valid outputs across all 72 cells, with zero duplicate planned IDs,
zero candidate-ID mutation, zero parser ambiguity, and passing seal, commit,
hardware, and model-revision provenance checks. If either model fails, the
campaign is not scaled.
