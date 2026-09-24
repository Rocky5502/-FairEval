import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.build_overleaf_bundle import (
    OPTIONAL_RESULT_FILES,
    REQUIRED_FILES,
    build_overleaf_bundle,
)


def test_build_overleaf_bundle_contains_required_anonymous_sources(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "build_paper_figures.py")],
        cwd=repo_root,
        check=True,
    )
    output = tmp_path / "fair_eval_overleaf.zip"
    result = build_overleaf_bundle(output_zip=output, include_available_results=False)
    assert output.is_file()
    assert result["files"] == len(REQUIRED_FILES)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert set(REQUIRED_FILES).issubset(names)
        assert "OVERLEAF_BUNDLE_MANIFEST.json" in names
        assert not any(name.startswith("../") or name.startswith("/") for name in names)
        main = archive.read("main.tex").decode("utf-8")
        assert r"\author{Anonymous Authors}" in main
        assert r"\institute{Anonymous Institution}" in main
        manifest = json.loads(archive.read("OVERLEAF_BUNDLE_MANIFEST.json"))
        assert manifest["double_blind"] is True
        assert manifest["entrypoint"] == "main.tex"
        assert manifest["empirical_numbers_traceable_to_audited_artifacts"] is True
        assert manifest["generated_result_artifacts_machine_derived"] is True
        assert manifest["prose_result_summaries_cross_checked"] is True
        assert set(manifest["files"]) == set(REQUIRED_FILES)


def test_generated_whitebox_table_is_an_optional_artifact_input():
    assert "generated/whitebox_summary_table.tex" in OPTIONAL_RESULT_FILES


def test_generated_fairsynth_figure_is_an_optional_artifact_input():
    assert "figures/fairsynth_hosted_effects.pdf" in OPTIONAL_RESULT_FILES


def test_hosted_pilot_operational_artifacts_are_optional_bundle_inputs():
    assert "generated/hosted_pilot_operational_table.tex" in OPTIONAL_RESULT_FILES
    assert "figures/hosted_pilot_blackbox_summary.pdf" in OPTIONAL_RESULT_FILES


def test_overleaf_bundle_rejects_zero_byte_optional_result(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "paper"
    paper = tmp_path / "paper"
    for name in REQUIRED_FILES:
        target = paper / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)

    bad = paper / "generated" / "v7_repetition_stability_table.tex"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"")

    with pytest.raises(ValueError, match="zero-byte generated result artifacts"):
        build_overleaf_bundle(
            paper_dir=paper,
            output_zip=tmp_path / "bad.zip",
            include_available_results=True,
        )
