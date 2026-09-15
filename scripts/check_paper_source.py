from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN = PAPER / "main.tex"
BIB = PAPER / "references.bib"


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text))


def _cite_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", text):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def _input_targets(text: str) -> list[Path]:
    targets = []
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        candidate = PAPER / name
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".tex")
        targets.append(candidate)
    return targets


def main() -> int:
    main_text = MAIN.read_text(encoding="utf-8")
    bib_text = BIB.read_text(encoding="utf-8")

    missing_cites = sorted(_cite_keys(main_text) - _bib_keys(bib_text))
    if missing_cites:
        raise SystemExit(f"missing bibliography keys: {missing_cites}")

    missing_inputs = [str(path.relative_to(ROOT)) for path in _input_targets(main_text) if not path.is_file()]
    if missing_inputs:
        raise SystemExit(f"missing LaTeX input files: {missing_inputs}")

    required_assets = [
        PAPER / "figures" / "faireval_framework.pdf",
        PAPER / "figures" / "faireval_conditions.pdf",
        PAPER / "related_work_table.tex",
    ]
    missing_assets = [str(path.relative_to(ROOT)) for path in required_assets if not path.is_file()]
    if missing_assets:
        raise SystemExit(f"missing manuscript assets: {missing_assets}")

    stale_phrases = [
        "DeepSeek V4 Flash, Qwen 3.8 Max",
        "DeepSeek V4 Flash,",
    ]
    stale = [phrase for phrase in stale_phrases if phrase in main_text]
    if stale:
        raise SystemExit(f"stale model-panel wording found: {stale}")

    if "\\author{Anonymous Authors}" not in main_text:
        raise SystemExit("double-blind author placeholder missing")
    if "Numerical findings are intentionally omitted" not in main_text:
        raise SystemExit("result-integrity statement missing from abstract")

    related = (PAPER / "related_work_table.tex").read_text(encoding="utf-8")
    if related.count("\\\\") < 16:
        raise SystemExit("related-work comparison table appears incomplete")
    if "do not imply empirical superiority" not in related:
        raise SystemExit("comparison-table interpretation guard missing")

    print(f"paper preflight: {len(_cite_keys(main_text))} citation keys resolved")
    print("paper preflight: LaTeX inputs and required vector assets present")
    print("paper preflight: double-blind and result-integrity guards present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
