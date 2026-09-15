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
    """Plot behavioral shift against preference-conditioned utility consequence.

    X = 1 - RBO@K, computed only for repetitions where both matched conditions
    produced valid rankings. Y = observed-demographic minus counterfactual nDCG.
    The zero-utility line is descriptive; no post-hoc harm threshold is invented.
    """
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


def _require_future_artifact(path: Path | None, *, rq: str) -> None:
    if path is None:
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"{rq} artifact is empty")
    raise NotImplementedError(
        f"{rq} result-figure schema is intentionally not guessed. Freeze the {rq} "
        "analysis artifact contract first, then implement its renderer."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate FairEval result PDFs strictly from frozen analysis artifacts"
    )
    parser.add_argument("--rq1-pairs", help="analysis/rq1_pairs.jsonl")
    parser.add_argument("--inference", help="analysis/inference.jsonl")
    parser.add_argument("--rq3-artifact", help="future frozen RQ3 robustness artifact")
    parser.add_argument("--rq4-artifact", help="future frozen RQ4 mitigation artifact")
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

    _require_future_artifact(None if args.rq3_artifact is None else Path(args.rq3_artifact), rq="RQ3")
    _require_future_artifact(None if args.rq4_artifact is None else Path(args.rq4_artifact), rq="RQ4")

    if not built and not args.rq3_artifact and not args.rq4_artifact:
        raise ValueError("supply at least one analysis artifact")
    print(json.dumps({"schema_version": "faireval-result-figures-v1", "built": built}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
