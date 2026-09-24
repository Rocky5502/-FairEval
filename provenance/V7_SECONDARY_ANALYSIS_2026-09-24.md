# FairEval V7 secondary analysis provenance — 2026-09-24

## Scope

This analysis is a **descriptive secondary analysis of the already frozen V7
local FairSynth results**. It makes no model calls, no hosted API calls, no GPU
inference, and introduces no new confirmatory hypothesis family.

The primary paired inference in
`results/analysis/fairsynth-v7-lean/inference.jsonl` remains unchanged.

## Frozen inputs

- user-condition SHA-256:
  `653a3eb04975b22975c5e8d40b861886d7ab8d18c3831c3f450d80adc5c4466b`
- identity-pairs SHA-256:
  `d1d997b51dd83d78c61216f77b186c23dd4965232901d8ac81cc0b53da36ac74`
- personality-pairs SHA-256:
  `7b6f6d212be7ed7a5344d906eccd61d9f5ca7e184c6649bb512b2b49edf59747`

The renderer/analysis implementation is
`scripts/analyze_v7_secondary.py`.

## Condition-level descriptive utility

### Phi-3.5-mini

| Context | Mean nDCG@10 | 95% descriptive bootstrap CI | Mean Recall@10 | Mean invalid rate |
|---|---:|---:|---:|---:|
| Preference only | 0.30368 | [0.27092, 0.33700] | 0.38417 | 0.00000 |
| Observed identity | 0.30502 | [0.26874, 0.34106] | 0.37583 | 0.00000 |
| Counterfactual identity | 0.29554 | [0.26266, 0.32937] | 0.36417 | 0.00000 |
| True synthetic OCEAN | 0.27270 | [0.24141, 0.30424] | 0.35333 | 0.00000 |
| Shuffled synthetic OCEAN | 0.28642 | [0.25412, 0.31915] | 0.37250 | 0.00000 |

### Qwen2.5-7B

| Context | Mean nDCG@10 | 95% descriptive bootstrap CI | Mean Recall@10 | Mean invalid rate |
|---|---:|---:|---:|---:|
| Preference only | 0.32115 | [0.28827, 0.35542] | 0.41500 | 0.03750 |
| Observed identity | 0.31125 | [0.28066, 0.34263] | 0.40500 | 0.04167 |
| Counterfactual identity | 0.31935 | [0.28860, 0.35048] | 0.41000 | 0.05417 |
| True synthetic OCEAN | 0.33068 | [0.29703, 0.36491] | 0.40333 | 0.05417 |
| Shuffled synthetic OCEAN | 0.32191 | [0.29011, 0.35505] | 0.40000 | 0.04583 |

These are descriptive context profiles, not a new inferential family.

## Paired nDCG distribution summaries

| Contrast | Model | Mean Δ | Median Δ | Q1 | Q3 | Positive | Negative | Zero |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Synthetic identity | Phi-3.5-mini | +0.00947 | 0.00000 | -0.01543 | 0.00000 | 24.17% | 30.00% | 45.83% |
| Synthetic identity | Qwen2.5-7B | -0.00809 | 0.00000 | -0.03437 | +0.03690 | 36.67% | 40.83% | 22.50% |
| Synthetic personality | Phi-3.5-mini | -0.01372 | 0.00000 | 0.00000 | 0.00000 | 16.67% | 22.50% | 60.83% |
| Synthetic personality | Qwen2.5-7B | +0.00877 | 0.00000 | -0.06190 | +0.07873 | 43.33% | 40.00% | 16.67% |

Interpretation: the near-zero aggregate effects are not universal invariance.
User-level effects can be mixed in sign even when the paired mean is small and
its confidence interval includes zero.

## Two-repetition stability

| Model | Valid repeat pairs | Mean RBO@10 | Mean Jaccard@10 | Exact order |
|---|---:|---:|---:|---:|
| Phi-3.5-mini | 720/720 (100.0%) | 0.955 | 0.863 | 61.9% |
| Qwen2.5-7B | 673/720 (93.5%) | 0.887 | 0.783 | 49.3% |

Only user-condition cells with two semantically valid rankings contribute to
RBO/Jaccard. The valid-pair rate is reported so semantic failures are not
conditioned away.

## Generated artifacts

A successful Windows V4 closeout generated non-empty:

- `results/analysis/fairsynth-v7-secondary/secondary_summary.json`
- `paper/figures/v7_secondary_profiles.pdf`
- `paper/generated/v7_repetition_stability_table.tex`

The final user-side V4 closeout also verified that these result artifacts were
non-empty inside the Overleaf ZIP. The analysis remains optional/descriptive in
the paper and never changes the frozen primary inference.
