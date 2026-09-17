from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MAIN_TABLES = ("rq12_main_table.tex", "rq34_main_table.tex")
COVERAGE_TABLE = "coverage_table.tex"
DATASET_LABELS = {
    "personality2018": "Personality 2018",
    "music_master_bfi2": "Music Master/BFI-2",
    "reasoner": "REASONER",
    "movielens_1m": "MovieLens-1M",
    "lastfm_1k": "Last.fm-1K",
    "mind": "MIND",
    "fairsynth360": "FairSynth-360",
}


def _normalize_main_table(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(r"\begin{table*}[t]", r"\begin{table}[t]")
    text = text.replace(r"\end{table*}", r"\end{table}")
    if r"\begin{table*}" in text or r"\end{table*}" in text:
        raise RuntimeError(f"{path}: unresolved table* token after LNCS normalization")
    if r"\begin{table}[t]" not in text or r"\end{table}" not in text:
        raise RuntimeError(f"{path}: expected ordinary LNCS table float after normalization")
    path.write_text(text, encoding="utf-8", newline="\n")


def _normalize_coverage_labels(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for dataset_id, display_name in DATASET_LABELS.items():
        text = text.replace(dataset_id, display_name)
        text = text.replace(dataset_id.replace("_", r"\_"), display_name)
    if "FairSynth-360" not in text:
        raise RuntimeError(f"{path}: FairSynth-360 display label missing after normalization")
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", default="paper/result_tables")
    known, _ = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    renderer = repo_root / "scripts" / "render_result_tables.py"
    command = [sys.executable, str(renderer), *sys.argv[1:]]
    subprocess.run(command, cwd=repo_root, check=True)

    output_dir = Path(known.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    for name in MAIN_TABLES:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"renderer did not create required main table: {path}")
        _normalize_main_table(path)

    coverage = output_dir / COVERAGE_TABLE
    if not coverage.is_file():
        raise FileNotFoundError(f"renderer did not create required coverage table: {coverage}")
    _normalize_coverage_labels(coverage)

    print(
        "LNCS result-table normalization: PASS "
        "(ordinary main-table floats + publication dataset labels)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
