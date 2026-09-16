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


def _require_tokens(path: Path, tokens: tuple[str, ...], *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    for token in tokens:
        if token not in text:
            raise SystemExit(f"{label} missing required contract token: {token}")


def main() -> int:
    missing = [name for name in REQUIRED if not (TABLE_DIR / name).is_file()]
    if missing:
        raise SystemExit(f"missing result-table contracts: {missing}")

    for name, tokens in REQUIRED.items():
        _require_tokens(TABLE_DIR / name, tokens, label=name)

    entrypoint = PAPER / "results_contract_table.tex"
    _require_tokens(
        entrypoint,
        (
            r"\input{result_tables/rq12_main_table}",
            r"\input{result_tables/rq34_main_table}",
        ),
        label="results entrypoint",
    )

    renderer = ROOT / "scripts" / "render_result_tables.py"
    _require_tokens(
        renderer,
        (
            "render_rq12_main",
            "render_rq34_main",
            "render_coverage",
            "render_trait_table",
            "render_fairsynth_table",
            "render_whitebox_table",
            "--rq4-prompting-summary-json",
            "final rendering requires all frozen artifacts",
            "confirmatory_p_values=false",
        ),
        label="result-table renderer",
    )

    _require_tokens(
        ROOT / "src" / "faireval" / "trait_analysis.py",
        ("true_vs_one_trait_counterfactual", "robustness_only"),
        label="RQ2 C5 analysis bridge",
    )
    _require_tokens(
        ROOT / "src" / "faireval" / "rq4_prompting.py",
        (
            "faireval-run-plan-v1",
            "rq4_identity_irrelevance_prompting",
            "identity_irrelevance_observed_vs_counterfactual",
            "confirmatory_p_values",
        ),
        label="RQ4 prompting analysis bridge",
    )
    _require_tokens(
        ROOT / "scripts" / "build_rq4_prompt_plan.py",
        ("run_plan_file_sha256", "planned_api_cells", "load_and_verify_plan"),
        label="RQ4 prompting plan compiler",
    )
    _require_tokens(
        ROOT / "scripts" / "analyze_rq4_prompting.py",
        ("prompting_summary.json", "confirmatory_p_value"),
        label="RQ4 prompting analyzer",
    )

    print("result-table preflight: 2 main + 4 compact contracts present")
    print("result-table preflight: main manuscript entrypoint uses the two main tables")
    print("result-table preflight: C5 trait analysis is wired")
    print("result-table preflight: all three RQ4 methods have executable evidence paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
