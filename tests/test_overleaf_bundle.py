import json
import zipfile
from pathlib import Path

from scripts.build_overleaf_bundle import (
    OPTIONAL_RESULT_FILES,
    REQUIRED_FILES,
    build_overleaf_bundle,
)


def test_build_overleaf_bundle_contains_required_anonymous_sources(tmp_path: Path):
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
        assert manifest["empirical_numbers_manually_entered"] is False
        assert set(manifest["files"]) == set(REQUIRED_FILES)


def test_generated_whitebox_table_is_an_optional_artifact_input():
    assert "generated/whitebox_summary_table.tex" in OPTIONAL_RESULT_FILES
