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

    # LNCS is a single-column layout. Main result contracts must not regress to
    # table* merely because the original renderer used a two-column-style float.
    for name in ("rq12_main_table.tex", "rq34_main_table.tex"):
        text = (TABLE_DIR / name).read_text(encoding="utf-8")
        if r"\begin{table*}" in text or r"\end{table*}" in text:
            raise SystemExit(f"{name} must use ordinary LNCS table floats, not table*")

    entrypoint = PAPER / "results_contract_table.tex"
    _require_tokens(
        entrypoint,
        ("generated/fairsynth_local_table.tex",),
        label="results entrypoint",
    )
    entrypoint_text = entrypoint.read_text(encoding="utf-8")
    for forbidden in (
        "generated/rq12_inference_table.tex",
        "generated/rq34_results_table.tex",
        "generated/fairsynth_hosted_table.tex",
        "generated/v7_execution_quality_table.tex",
        "generated/whitebox_summary_table.tex",
    ):
        if forbidden in entrypoint_text:
            raise SystemExit(
                f"page-limited ECIR entrypoint must not auto-typeset auxiliary table: {forbidden}"
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

    # Real-result generation must go through the LNCS wrapper so a future
    # renderer invocation cannot recreate table* in the single-column paper or
    # regress canonical dataset IDs into unpolished paper-facing labels.
    _require_tokens(
        ROOT / "scripts" / "render_result_tables_lncs.py",
        (
            "render_result_tables.py",
            "MAIN_TABLES",
            "table*",
            "_normalize_coverage_labels",
            "FairSynth-360",
            "LNCS result-table normalization: PASS",
        ),
        label="LNCS result-table wrapper",
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

    print("result-table preflight: legacy contracts retained; ECIR entrypoint typesets only the primary paired-effect table")
    print("result-table preflight: main result contracts use ordinary LNCS table floats")
    print("result-table preflight: canonical real-result rendering uses the LNCS wrapper")
    print("result-table preflight: publication dataset labels are normalized")
    print("result-table preflight: C5 trait analysis is wired")
    print("result-table preflight: all three RQ4 methods have executable evidence paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
