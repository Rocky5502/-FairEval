from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MAIN_TABLES = ("rq12_main_table.tex", "rq34_main_table.tex")


def _normalize_main_table(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(r"\begin{table*}[t]", r"\begin{table}[t]")
    text = text.replace(r"\end{table*}", r"\end{table}")
    if r"\begin{table*}" in text or r"\end{table*}" in text:
        raise RuntimeError(f"{path}: unresolved table* token after LNCS normalization")
    if r"\begin{table}[t]" not in text or r"\end{table}" not in text:
        raise RuntimeError(f"{path}: expected ordinary LNCS table float after normalization")
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

    print(
        "LNCS result-table normalization: PASS "
        "(RQ1-RQ2 and RQ3-RQ4 use ordinary table floats)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
