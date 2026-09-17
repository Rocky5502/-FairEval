from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

RESULT_TABLE_FILES = (
    "result_tables/rq12_main_table.tex",
    "result_tables/rq34_main_table.tex",
    "result_tables/coverage_table.tex",
    "result_tables/trait_ablation_table.tex",
    "result_tables/fairsynth_table.tex",
    "result_tables/whitebox_table.tex",
)

REQUIRED_FILES = (
    "main.tex",
    "benchmark_model_table.tex",
    "rq_design_table.tex",
    "results_contract_table.tex",
    "related_work_table.tex",
    "references.bib",
    "references_extra.bib",
    "OVERLEAF_SYNC.md",
    "figures/faireval_framework.pdf",
    "figures/faireval_conditions.pdf",
    "figures/faireval_evaluation_pipeline.pdf",
    *RESULT_TABLE_FILES,
)

OPTIONAL_RESULT_FILES = (
    # Artifact-generated LaTeX tables. These are the authoritative numerical
    # paper inputs whenever present and must travel with the Overleaf archive.
    "generated/rq12_inference_table.tex",
    "generated/rq34_results_table.tex",
    "generated/fairsynth_hosted_table.tex",
    # Artifact-generated result figures.
    "figures/rq1_quadrant.pdf",
    "figures/rq2_personality_forest.pdf",
    "figures/rq3_variance.pdf",
    "figures/rq4_pareto.pdf",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_overleaf_bundle(
    *,
    paper_dir: Path = PAPER,
    output_zip: Path,
    include_available_results: bool = True,
) -> dict[str, Any]:
    """Create a deterministic upload-ready archive from canonical paper sources.

    The ZIP deliberately excludes repository code, raw datasets, run logs,
    secrets, and author-identifying metadata. Generated result tables/figures are
    included only when they already exist as artifact-derived paper inputs; the
    manuscript otherwise compiles registered placeholders. This means the ZIP
    cannot silently drop numerical tables that were already rendered locally.
    """
    missing = [name for name in REQUIRED_FILES if not (paper_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"paper directory is missing required Overleaf files: {missing}")

    main_text = (paper_dir / "main.tex").read_text(encoding="utf-8")
    for token in (
        r"\author{Anonymous Authors}",
        r"\authorrunning{Anonymous Authors}",
        r"\institute{Anonymous Institution}",
    ):
        if token not in main_text:
            raise ValueError(f"double-blind Overleaf bundle guard missing: {token}")

    results_entrypoint = (paper_dir / "results_contract_table.tex").read_text(encoding="utf-8")
    for token in (
        r"\input{result_tables/rq12_main_table}",
        r"\input{result_tables/rq34_main_table}",
        "generated/fairsynth_hosted_table.tex",
    ):
        if token not in results_entrypoint:
            raise ValueError(f"results entrypoint missing result contract: {token}")

    selected = list(REQUIRED_FILES)
    if include_available_results:
        selected.extend(name for name in OPTIONAL_RESULT_FILES if (paper_dir / name).is_file())
    selected = sorted(set(selected))

    manifest_files = {
        name: {
            "sha256": _sha256(paper_dir / name),
            "bytes": (paper_dir / name).stat().st_size,
        }
        for name in selected
    }
    generated_tables = sorted(
        name
        for name in OPTIONAL_RESULT_FILES
        if name.startswith("generated/") and name in selected
    )
    generated_figures = sorted(
        name
        for name in OPTIONAL_RESULT_FILES
        if name.startswith("figures/") and name in selected
    )
    manifest = {
        "schema_version": "faireval-overleaf-bundle-v3",
        "entrypoint": "main.tex",
        "double_blind": True,
        "canonical_source": "paper/ on ecir-2027-redesign",
        "empirical_numbers_manually_entered": False,
        "result_table_contract": {
            "fallback_main_tables": [
                "result_tables/rq12_main_table.tex",
                "result_tables/rq34_main_table.tex",
            ],
            "compact_supporting_tables": list(RESULT_TABLE_FILES[2:]),
            "artifact_generated_tables_included": generated_tables,
        },
        "artifact_generated_figures_included": generated_figures,
        "files": manifest_files,
    }

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in selected:
            archive.write(paper_dir / name, arcname=name)
        archive.writestr(
            "OVERLEAF_BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    return {
        "schema_version": "faireval-overleaf-bundle-build-v3",
        "output": str(output_zip),
        "files": len(selected),
        "zip_sha256": _sha256(output_zip),
        "included_result_table_contracts": list(RESULT_TABLE_FILES),
        "included_generated_result_tables": generated_tables,
        "included_generated_result_figures": generated_figures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an anonymous upload-ready FairEval ECIR 2027 Overleaf ZIP"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "FairEval_ECIR2027_Overleaf.zip",
    )
    parser.add_argument(
        "--exclude-results",
        action="store_true",
        help="Do not include generated result LaTeX/PDF artifacts even if they exist.",
    )
    args = parser.parse_args()
    result = build_overleaf_bundle(
        output_zip=args.output,
        include_available_results=not args.exclude_results,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
