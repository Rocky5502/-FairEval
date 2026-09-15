from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


FAMILY_ORDER = ("openai", "anthropic", "google", "deepseek", "qwen", "meta")
FAMILY_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "meta": "Llama",
}
FACTOR_ORDER = ("prompt", "cue", "candidate_order", "cutoff", "stochasticity")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _configure_pdf_fonts() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def build_rq1_quadrant(rq1_pairs: Path, output: Path) -> None:
    """Plot behavioral shift against preference-conditioned utility consequence."""
    rows = [
        row
        for row in _read_jsonl(rq1_pairs)
        if row.get("rq") == "RQ1"
        and row.get("contrast") == "observed_vs_demographic_counterfactual"
        and row.get("mean_one_minus_rbo") is not None
    ]
    if not rows:
        raise ValueError("RQ1 artifact has no valid matched-ranking diagnostics")

    _configure_pdf_fonts()
    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    markers = ("o", "s", "^", "D", "P", "X")
    for marker, family in zip(markers, FAMILY_ORDER, strict=True):
        family_rows = [row for row in rows if str(row.get("model_family")) == family]
        if not family_rows:
            continue
        x = [float(row["mean_one_minus_rbo"]) for row in family_rows]
        y = [float(row["delta_ndcg"]) for row in family_rows]
        ax.scatter(x, y, marker=marker, alpha=0.55, s=24, label=FAMILY_LABELS[family])

    ax.axhline(0.0, linewidth=0.9, linestyle="--")
    ax.set_xlabel("Behavioral ranking change (1 - RBO@K)")
    ax.set_ylabel("Counterfactual utility gap: nDCG(observed) - nDCG(counterfactual)")
    ax.set_title("RQ1: ranking change is not itself a fairness verdict", loc="left", fontweight="bold")
    ax.text(
        0.99,
        0.97,
        "Above 0: observed context has higher held-out utility\nBelow 0: counterfactual context has higher held-out utility",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    ax.grid(axis="both", linewidth=0.35, alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=7.2, loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_rq2_forest(inference_jsonl: Path, output: Path) -> None:
    """Forest plot for true measured personality vs shuffled-profile PVA."""
    rows = [
        row
        for row in _read_jsonl(inference_jsonl)
        if row.get("rq") == "RQ2"
        and row.get("contrast") == "true_vs_shuffled_personality"
        and row.get("metric") == "ndcg"
    ]
    if not rows:
        raise ValueError("inference artifact has no RQ2 true-vs-shuffled nDCG rows")

    order = {family: index for index, family in enumerate(FAMILY_ORDER)}
    rows.sort(key=lambda row: (str(row["dataset"]), order.get(str(row["model_family"]), 999)))
    labels = [f"{row['dataset']} / {FAMILY_LABELS.get(str(row['model_family']), row['model_family'])}" for row in rows]
    effects = [float(row["mean_paired_difference"]) for row in rows]
    low = [float(row["bootstrap_ci_low"]) for row in rows]
    high = [float(row["bootstrap_ci_high"]) for row in rows]
    left_err = [effect - lo for effect, lo in zip(effects, low, strict=True)]
    right_err = [hi - effect for effect, hi in zip(effects, high, strict=True)]

    _configure_pdf_fonts()
    height = max(3.7, 0.24 * len(rows) + 1.35)
    fig, ax = plt.subplots(figsize=(6.8, height))
    y = list(range(len(rows)))
    ax.errorbar(
        effects,
        y,
        xerr=[left_err, right_err],
        fmt="o",
        markersize=4.2,
        capsize=2.2,
        linewidth=0.9,
    )
    ax.axvline(0.0, linewidth=0.9, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("PVA in nDCG@10: true measured profile - shuffled profile")
    ax.set_title("RQ2: user-specific value of measured personality", loc="left", fontweight="bold")
    ax.grid(axis="x", linewidth=0.35, alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_rq3_variance(rq3_summary: Path, output: Path) -> None:
    """Visualize one-factor-at-a-time reliability variation without pooling it away."""
    rows = [
        row
        for row in _read_jsonl(rq3_summary)
        if row.get("schema_version") == "faireval-rq3-variation-summary-v1"
        and row.get("metric") == "ndcg"
        and str(row.get("factor")) in FACTOR_ORDER
    ]
    if not rows:
        raise ValueError("RQ3 summary contains no nDCG reliability rows")

    _configure_pdf_fonts()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    family_marker = dict(zip(FAMILY_ORDER, ("o", "s", "^", "D", "P", "X"), strict=True))
    factor_x = {factor: idx for idx, factor in enumerate(FACTOR_ORDER)}
    offsets = {family: (idx - 2.5) * 0.035 for idx, family in enumerate(FAMILY_ORDER)}

    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if str(row.get("model_family")) == family]
        if not family_rows:
            continue
        x = [factor_x[str(row["factor"])] + offsets[family] for row in family_rows]
        y = [float(row["mean_within_user_sd"]) for row in family_rows]
        ax.scatter(
            x,
            y,
            marker=family_marker[family],
            alpha=0.62,
            s=26,
            label=FAMILY_LABELS[family],
        )

    ax.set_xticks(
        list(range(len(FACTOR_ORDER))),
        ["Prompt", "Cue", "Candidate\norder", "K", "Generation"],
    )
    ax.set_ylabel("Mean within-user SD of nDCG@10")
    ax.set_title("RQ3: reliability under one-factor-at-a-time perturbations", loc="left", fontweight="bold")
    ax.grid(axis="y", linewidth=0.35, alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=7.2, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_rq4_pareto(rq4_artifact: Path, output: Path) -> None:
    """Render the validation frontier and frozen test operating point for contextual PAIR.

    The renderer consumes only ``faireval-rq4-pair-artifact-v1``. It cannot select
    a hyperparameter point itself; the selected point must already be frozen by
    the validation-only analyzer. Lower absolute CUG is better (y axis); higher
    end-to-end identity-conditioned nDCG is better (x axis).
    """
    artifact = _read_json_object(rq4_artifact)
    if artifact.get("schema_version") != "faireval-rq4-pair-artifact-v1":
        raise ValueError("unsupported RQ4 artifact schema")
    if artifact.get("selection_used_test_outcomes") is not False:
        raise ValueError("RQ4 artifact indicates test outcomes influenced selection")
    if artifact.get("per_model_or_dataset_tuning") is not False:
        raise ValueError("RQ4 artifact indicates per-model/per-dataset tuning")

    frontier = artifact.get("validation_frontier")
    operating = artifact.get("operating_point")
    test_summary = artifact.get("test_summary")
    if not isinstance(frontier, list) or not frontier:
        raise ValueError("RQ4 artifact has no validation frontier")
    if not isinstance(operating, dict) or not isinstance(test_summary, dict):
        raise ValueError("RQ4 artifact lacks frozen operating point/test summary")

    utility_floor = float(operating.get("utility_floor_ratio", 0.95))
    eligible = [
        row
        for row in frontier
        if row.get("pair_abs_cug_ndcg_on_available") is not None
        and float(row.get("utility_retention", 0.0)) >= utility_floor
    ]
    ineligible = [
        row
        for row in frontier
        if row.get("pair_abs_cug_ndcg_on_available") is not None
        and float(row.get("utility_retention", 0.0)) < utility_floor
    ]
    if not eligible:
        raise ValueError("RQ4 artifact has no validation point satisfying the frozen utility floor")

    _configure_pdf_fonts()
    fig, ax = plt.subplots(figsize=(6.9, 4.25))
    if ineligible:
        ax.scatter(
            [float(row["pair_identity_ndcg_mean_system"]) for row in ineligible],
            [float(row["pair_abs_cug_ndcg_on_available"]) for row in ineligible],
            marker="x",
            alpha=0.45,
            s=30,
            label=f"Validation: below {utility_floor:.0%} utility floor",
        )
    ax.scatter(
        [float(row["pair_identity_ndcg_mean_system"]) for row in eligible],
        [float(row["pair_abs_cug_ndcg_on_available"]) for row in eligible],
        marker="o",
        alpha=0.65,
        s=32,
        label="Validation: eligible grid points",
    )

    chosen_validation = operating.get("validation_summary")
    if not isinstance(chosen_validation, dict):
        raise ValueError("RQ4 operating point lacks validation summary")
    ax.scatter(
        [float(chosen_validation["pair_identity_ndcg_mean_system"])],
        [float(chosen_validation["pair_abs_cug_ndcg_on_available"])],
        marker="*",
        s=120,
        label="Frozen validation operating point",
        zorder=4,
    )

    test_cug = test_summary.get("pair_abs_cug_ndcg_on_available")
    if test_cug is None:
        raise ValueError("RQ4 frozen test point has no available counterfactual pairs")
    ax.scatter(
        [float(test_summary["pair_identity_ndcg_mean_system"])],
        [float(test_cug)],
        marker="D",
        s=52,
        label="Held-out test result at frozen point",
        zorder=5,
    )

    alpha = float(operating["alpha"])
    lam = float(operating["lambda_instability"])
    ax.set_xlabel("End-to-end identity-conditioned nDCG@10 (higher is better)")
    ax.set_ylabel("Absolute CUG in nDCG@10 (lower is better)")
    ax.set_title("RQ4: contextual PAIR utility-fairness frontier", loc="left", fontweight="bold")
    ax.text(
        0.99,
        0.97,
        f"Frozen on validation only: alpha={alpha:g}, lambda={lam:g}\n"
        f"utility retention floor={utility_floor:.0%}; test never selects",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    ax.grid(axis="both", linewidth=0.35, alpha=0.25)
    ax.legend(frameon=False, fontsize=7.0, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate FairEval result PDFs strictly from frozen analysis artifacts"
    )
    parser.add_argument("--rq1-pairs", help="analysis/rq1_pairs.jsonl")
    parser.add_argument("--inference", help="analysis/inference.jsonl")
    parser.add_argument("--rq3-artifact", help="RQ3 variation summary JSONL")
    parser.add_argument("--rq4-artifact", help="RQ4 contextual PAIR artifact JSON")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    built: list[str] = []
    if args.rq1_pairs:
        path = output_dir / "rq1_quadrant.pdf"
        build_rq1_quadrant(Path(args.rq1_pairs), path)
        built.append(str(path))
    if args.inference:
        path = output_dir / "rq2_personality_forest.pdf"
        build_rq2_forest(Path(args.inference), path)
        built.append(str(path))
    if args.rq3_artifact:
        path = output_dir / "rq3_variance.pdf"
        build_rq3_variance(Path(args.rq3_artifact), path)
        built.append(str(path))
    if args.rq4_artifact:
        path = output_dir / "rq4_pareto.pdf"
        build_rq4_pareto(Path(args.rq4_artifact), path)
        built.append(str(path))

    if not built:
        raise ValueError("supply at least one analysis artifact")
    print(json.dumps({"schema_version": "faireval-result-figures-v2", "built": built}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
