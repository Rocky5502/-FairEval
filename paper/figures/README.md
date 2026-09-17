# FairEval figure source contract

The paper uses two different figure classes and deliberately treats them differently.

## Conceptual / non-result figures

The canonical source is `scripts/build_paper_figures.py`. One command regenerates each conceptual figure in two formats:

```bash
python scripts/build_paper_figures.py
```

Outputs:

- `faireval_framework.pdf` + `faireval_framework.svg`
- `faireval_conditions.pdf` + `faireval_conditions.svg`
- `faireval_evaluation_pipeline.pdf` + `faireval_evaluation_pipeline.svg`

The PDF is the publication asset. The SVG is the fully editable companion: shapes remain vector objects and `svg.fonttype = none` preserves text as text rather than paths. The SVG companions are generated/ignored in Git so rebuilding figures does not dirty a sealed experiment checkout; `scripts/build_overleaf_bundle.py` requires and bundles the SVGs after figure generation.

Figure 1 is an original FairEval motivating example. Its visual logic begins from the neutral-versus-sensitive recommendation comparison used in FaiRLLM, but the interpretation is intentionally different: FairEval treats list change as sensitivity, not as a fairness verdict, and requires preference-conditioned consequence evidence before interpreting a potential disparity.

## Result figures

`rq1_quadrant.pdf`, `rq2_personality_forest.pdf`, `rq3_variance.pdf`, and `rq4_pareto.pdf` are generated only by the audited analysis pipeline (`scripts/build_result_figures.py`). Their empirical geometry and values must not be manually edited. If styling needs to change, edit the renderer and regenerate from the same audited artifact.

## Visual style

Conceptual figures use a restrained scientific palette, rounded grouping only where it conveys experimental structure, lightweight line icons, no vendor logos, no decorative gradients, and no empirical numbers. Labels distinguish confirmatory evidence, robustness controls, synthetic sanity checks, and local-only diagnostics. Tables remain native LaTeX/booktabs so every table is directly editable and consistent with the LNCS manuscript.
