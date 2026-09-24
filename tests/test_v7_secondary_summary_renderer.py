from pathlib import Path

from scripts.render_v7_secondary_summary import _load, render_figure, render_stability_table


def test_frozen_secondary_summary_renders(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    summary = _load(
        root / "results" / "analysis" / "fairsynth-v7-secondary" / "secondary_summary.json"
    )
    figure = tmp_path / "secondary.pdf"
    render_figure(summary, figure)
    assert figure.is_file()
    assert figure.stat().st_size > 0

    table = render_stability_table(summary)
    assert "720/720" in table
    assert "673/720" in table
    assert "0.955" in table
    assert "0.887" in table
