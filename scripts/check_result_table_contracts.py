from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TABLE_DIR = PAPER / "result_tables"

REQUIRED = {
    "rq12_main_table.tex": (
        "tab:main-rq12-results",
        "Qwen2.5-7B local",
        "Phi-3.5-mini local",
        "RQ1 MovieLens-1M",
        "RQ2 REASONER",
    ),
    "rq34_main_table.tex": (
        "tab:main-rq34-results",
        "Task wording",
        "Identity-irrelevance prompting",
        "Contextual PAIR",
        "95\\%",
    ),
    "coverage_table.tex": (
        "tab:coverage-results",
        "FairSynth-360",
        "Invalid rate",
    ),
    "trait_ablation_table.tex": (
        "tab:trait-ablation-results",
        "Openness",
        "Neuroticism",
    ),
    "fairsynth_table.tex": (
        "tab:fairsynth-results",
        "Identity sanity / hosted",
        "Synthetic personality sanity / local",
    ),
    "whitebox_table.tex": (
        "tab:whitebox-results",
        "uncalibrated",
        "Token NLL",
    ),
}


def main() -> int:
    missing = [name for name in REQUIRED if not (TABLE_DIR / name).is_file()]
    if missing:
        raise SystemExit(f"missing result-table contracts: {missing}")

    for name, tokens in REQUIRED.items():
        text = (TABLE_DIR / name).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                raise SystemExit(f"{name} missing required contract token: {token}")

    entrypoint = (PAPER / "results_contract_table.tex").read_text(encoding="utf-8")
    expected_inputs = (
        r"\input{result_tables/rq12_main_table}",
        r"\input{result_tables/rq34_main_table}",
    )
    for token in expected_inputs:
        if token not in entrypoint:
            raise SystemExit(f"results entrypoint missing {token}")

    renderer = (ROOT / "scripts" / "render_result_tables.py").read_text(encoding="utf-8")
    for token in (
        "render_rq12_main",
        "render_rq34_main",
        "render_coverage",
        "render_trait_table",
        "render_fairsynth_table",
        "render_whitebox_table",
        "final rendering requires all frozen artifacts",
    ):
        if token not in renderer:
            raise SystemExit(f"result-table renderer missing contract token: {token}")

    trait_module = (ROOT / "src" / "faireval" / "trait_analysis.py").read_text(encoding="utf-8")
    if "true_vs_one_trait_counterfactual" not in trait_module:
        raise SystemExit("RQ2 C5 analysis bridge is missing")

    print("result-table preflight: 2 main + 4 compact contracts present")
    print("result-table preflight: main manuscript entrypoint uses the two main tables")
    print("result-table preflight: renderer and C5 trait analysis bridge are wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
