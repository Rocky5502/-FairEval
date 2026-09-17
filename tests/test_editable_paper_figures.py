from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_concept_figure_builder_emits_pdf_and_editable_svg() -> None:
    text = (ROOT / "scripts" / "build_paper_figures.py").read_text(encoding="utf-8")
    assert '"svg.fonttype": "none"' in text
    assert 'fig.savefig(pdf' in text
    assert 'fig.savefig(svg' in text
    for stem in (
        "faireval_framework",
        "faireval_conditions",
        "faireval_evaluation_pipeline",
    ):
        assert f'save(fig, "{stem}")' in text


def test_overleaf_bundle_requires_editable_concept_sources() -> None:
    text = (ROOT / "scripts" / "build_overleaf_bundle.py").read_text(encoding="utf-8")
    for name in (
        "figures/faireval_framework.svg",
        "figures/faireval_conditions.svg",
        "figures/faireval_evaluation_pipeline.svg",
    ):
        assert name in text
    assert "artifact-generated; do not manually edit empirical geometry or values" in text


def test_result_figures_remain_artifact_generated() -> None:
    text = (ROOT / "paper" / "figures" / "README.md").read_text(encoding="utf-8")
    assert "must not be manually edited" in text
    assert "scripts/build_result_figures.py" in text
