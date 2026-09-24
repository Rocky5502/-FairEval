from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "phi35_local": "Phi-3.5-mini",
    "qwen25_local": "Qwen2.5-7B",
}
MODEL_COLORS = {
    "phi35_local": "#0F766E",
    "qwen25_local": "#7C3AED",
}
CONDITION_ORDER = (
    "preference_only",
    "observed_identity",
    "counterfactual_identity",
    "true_ocean",
    "shuffled_ocean",
)
CONDITION_LABELS = {
    "preference_only": "Preference\nonly",
    "observed_identity": "Observed\nidentity",
    "counterfactual_identity": "Counterfactual\nidentity",
    "true_ocean": "True\nOCEAN",
    "shuffled_ocean": "Shuffled\nOCEAN",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "faireval-v7-secondary-analysis-v1":
        raise ValueError("unexpected V7 secondary summary schema")
    if value.get("new_model_or_api_calls") is not False:
        raise ValueError("secondary summary must certify zero new model/API calls")
    if value.get("new_confirmatory_hypothesis_family") is not False:
        raise ValueError("secondary summary must remain descriptive only")
    return value


def _condition_row(summary: dict[str, Any], model: str, condition: str) -> dict[str, Any]:
    for row in summary["condition_summary"]:
        if row["model_family"] == model and row["condition_group"] == condition:
            return row
    raise KeyError((model, condition))


def _pair_row(summary: dict[str, Any], contrast: str, model: str) -> dict[str, Any]:
    for row in summary["pair_distribution_summary"]:
        if row["contrast"] == contrast and row["model_family"] == model:
            return row
    raise KeyError((contrast, model))


def render_figure(summary: dict[str, Any], output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": "#64748B",
        "axes.labelcolor": "#172033",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "text.color": "#172033",
    })

    fig, (ax_profile, ax_effect) = plt.subplots(
        1, 2, figsize=(6.75, 2.75), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    x = np.arange(len(CONDITION_ORDER))
    for model in ("phi35_local", "qwen25_local"):
        rows = [_condition_row(summary, model, condition) for condition in CONDITION_ORDER]
        means = np.asarray([float(row["ndcg_mean"]) for row in rows])
        lower = means - np.asarray([float(row["ndcg_ci_low"]) for row in rows])
        upper = np.asarray([float(row["ndcg_ci_high"]) for row in rows]) - means
        ax_profile.errorbar(
            x,
            means,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.35,
            capsize=2.5,
            markersize=4.0,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )

    ax_profile.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax_profile.set_ylabel("Mean nDCG@10")
    ax_profile.set_title("(a) Utility by controlled context", loc="left", fontsize=8.5, fontweight="bold")
    ax_profile.grid(axis="y", linewidth=0.45, alpha=0.28)
    ax_profile.spines[["top", "right"]].set_visible(False)
    ax_profile.legend(frameon=False, fontsize=7.0, loc="best")

    groups = [
        ("Identity / Phi", "synthetic_identity", "phi35_local"),
        ("Identity / Qwen", "synthetic_identity", "qwen25_local"),
        ("Personality / Phi", "synthetic_personality", "phi35_local"),
        ("Personality / Qwen", "synthetic_personality", "qwen25_local"),
    ]
    y = np.arange(len(groups))[::-1]
    for ypos, (label, contrast, model) in zip(y, groups, strict=True):
        row = _pair_row(summary, contrast, model)
        q1 = float(row["q1_delta_ndcg"])
        q3 = float(row["q3_delta_ndcg"])
        median = float(row["median_delta_ndcg"])
        mean = float(row["mean_delta_ndcg"])
        color = MODEL_COLORS[model]
        ax_effect.hlines(ypos, q1, q3, color=color, linewidth=5.0, alpha=0.35)
        ax_effect.plot([q1, q3], [ypos, ypos], "|", color=color, markersize=9, markeredgewidth=1.1)
        ax_effect.plot(median, ypos, "o", color="#111827", markersize=3.5, label=None)
        ax_effect.plot(mean, ypos, "D", color=color, markersize=4.0)
        pos = 100.0 * float(row["positive_fraction"])
        neg = 100.0 * float(row["negative_fraction"])
        zero = 100.0 * float(row["zero_fraction"])
        ax_effect.text(
            0.99,
            ypos,
            f"+{pos:.0f}% / -{neg:.0f}% / 0={zero:.0f}%",
            transform=ax_effect.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.0,
            color="#475569",
        )

    ax_effect.axvline(0.0, linestyle="--", linewidth=0.8, color="#475569")
    ax_effect.set_yticks(y, [g[0] for g in groups])
    ax_effect.set_xlabel("Paired ΔnDCG@10")
    ax_effect.set_title("(b) Heterogeneous paired effects", loc="left", fontsize=8.5, fontweight="bold")
    ax_effect.grid(axis="x", linewidth=0.45, alpha=0.28)
    ax_effect.spines[["top", "right"]].set_visible(False)
    ax_effect.text(
        0.01, 0.01,
        "thick bar = IQR; circle = median; diamond = mean",
        transform=ax_effect.transAxes,
        ha="left", va="bottom", fontsize=5.8, color="#64748B",
    )

    fig.text(
        0.5, 0.006,
        "Descriptive secondary analysis of frozen V7 artifacts; no new model/API calls and no new confirmatory hypothesis family.",
        ha="center", va="bottom", fontsize=6.2, color="#64748B",
    )
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 1.0), w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_stability_table(summary: dict[str, Any]) -> str:
    by_model = {
        str(row["model_family"]): row for row in summary["repetition_stability_summary"]
    }
    body = []
    for model in ("phi35_local", "qwen25_local"):
        row = by_model[model]
        body.append(
            "{} & {}/{} ({:.1f}\\%) & {:.3f} & {:.3f} & {:.1f}\\% \\\\".format(
                MODEL_LABELS[model],
                int(row["valid_repeat_pairs"]),
                int(row["user_condition_cells"]),
                100.0 * float(row["valid_repeat_pair_rate"]),
                float(row["mean_rbo_at_10"]),
                float(row["mean_jaccard_at_10"]),
                100.0 * float(row["exact_order_fraction"]),
            )
        )
    return "\n".join([
        "% AUTO-GENERATED from frozen V7 secondary_summary.json; DO NOT EDIT.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Two-repetition stability in the completed V7 local run. RBO/Jaccard use only cells with two semantically valid rankings; the valid-pair rate keeps persistent failures visible.}",
        "\\label{tab:v7-repeat-stability}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & Valid repeat pairs & Mean RBO@10 & Mean Jaccard@10 & Exact order \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="results/analysis/fairsynth-v7-secondary/secondary_summary.json",
    )
    parser.add_argument(
        "--figure",
        default="paper/figures/v7_secondary_profiles.pdf",
    )
    parser.add_argument(
        "--stability-table",
        default="paper/generated/v7_repetition_stability_table.tex",
    )
    args = parser.parse_args()

    summary = _load(Path(args.summary))
    render_figure(summary, Path(args.figure))
    table_path = Path(args.stability_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(render_stability_table(summary), encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "summary": args.summary,
        "figure": args.figure,
        "stability_table": args.stability_table,
        "new_model_or_api_calls": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
