from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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
FAMILY_COLORS = {
    "openai": "#D97706",
    "anthropic": "#DB2777",
    "google": "#059669",
    "deepseek": "#2563EB",
    "qwen": "#7C3AED",
    "meta": "#CA8A04",
}
MODEL_IDS = {
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-sonnet-5",
    "google": "gemini-3.8-flash",
    "deepseek": "deepseek-v4.1-flash",
    "qwen": "qwen3.8-max",
    "meta": "llama-4-maverick",
}
ID_CONTRAST = "observed_identity_vs_mean_counterfactual_identity"
P_CONTRAST = "true_synthetic_ocean_vs_shuffled_profile"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _fmt(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _p(value: object) -> str:
    x = float(value)
    return "$<.001$" if x < 0.001 else f"{x:.3f}"


def _index_inference(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model_family"]), str(row["contrast"]), str(row["metric"]))
        if key in out:
            raise ValueError(f"duplicate hosted inference row: {key}")
        out[key] = row
    return out


def _effect(row: dict[str, Any]) -> str:
    return "{} [{}, {}]".format(
        _fmt(row["mean_paired_difference"]),
        _fmt(row["bootstrap_ci_low"]),
        _fmt(row["bootstrap_ci_high"]),
    )


def render_table(inference: list[dict[str, Any]]) -> str:
    idx = _index_inference(inference)
    body: list[str] = []
    for family in FAMILY_ORDER:
        for label, contrast in (("Identity", ID_CONTRAST), ("Personality", P_CONTRAST)):
            ndcg = idx[(family, contrast, "ndcg")]
            recall = idx[(family, contrast, "recall")]
            if int(ndcg["n_users"]) != int(recall["n_users"]):
                raise ValueError(f"n_users mismatch for {family}/{contrast}")
            body.append(
                "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                    FAMILY_LABELS[family],
                    label,
                    int(ndcg["n_users"]),
                    _effect(ndcg),
                    _p(ndcg["holm_adjusted_p"]),
                    _effect(recall),
                    _p(recall["holm_adjusted_p"]),
                )
            )

    return "\n".join([
        r"% AUTO-GENERATED from the sealed hosted paired extension; DO NOT EDIT.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Hosted FairSynth paired effects. Identity compares the observed meaningless A/B/C label with the mean of its alternatives; personality compares true with shuffled synthetic OCEAN. Values are paired mean differences [95\% bootstrap CI] with Holm-adjusted sign-flip $p_H$. Identity is a controlled fairness-sanity test, not a claim about real demographic fairness.}",
        r"\label{tab:hosted-fairsynth-results}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrcccc}",
        r"\toprule",
        r"Model & Contrast & $N$ & $\Delta$nDCG@10 [95\% CI] & $p_H$ & $\Delta$Recall@10 [95\% CI] & $p_H$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])


def _ci_contains_zero(row: dict[str, Any]) -> bool:
    return float(row["bootstrap_ci_low"]) <= 0.0 <= float(row["bootstrap_ci_high"])


def render_summary(inference: list[dict[str, Any]]) -> str:
    idx = _index_inference(inference)
    identity = [idx[(family, ID_CONTRAST, "ndcg")] for family in FAMILY_ORDER]
    personality = [idx[(family, P_CONTRAST, "ndcg")] for family in FAMILY_ORDER]

    n_values = {int(row["n_users"]) for row in identity + personality}
    if len(n_values) != 1:
        raise ValueError(f"hosted paired extension has unequal N: {sorted(n_values)}")
    n = next(iter(n_values))

    id_means = [float(row["mean_paired_difference"]) for row in identity]
    p_means = [float(row["mean_paired_difference"]) for row in personality]
    id_zero = sum(_ci_contains_zero(row) for row in identity)
    p_zero = sum(_ci_contains_zero(row) for row in personality)
    id_sig = [
        FAMILY_LABELS[family]
        for family, row in zip(FAMILY_ORDER, identity, strict=True)
        if float(row["holm_adjusted_p"]) < 0.05
    ]
    p_sig = [
        FAMILY_LABELS[family]
        for family, row in zip(FAMILY_ORDER, personality, strict=True)
        if float(row["holm_adjusted_p"]) < 0.05
    ]

    def sig_text(values: list[str]) -> str:
        return "none" if not values else ", ".join(values)

    return (
        "Across the six hosted families ($N={}$ paired users per family), "
        "the controlled identity $\\Delta$nDCG@10 ranges from {:.3f} to {:.3f}; "
        "{}/6 bootstrap intervals contain zero and Holm-adjusted $p_H<.05$ for {}. "
        "For RQ2, true-versus-shuffled synthetic OCEAN $\\Delta$nDCG@10 ranges "
        "from {:.3f} to {:.3f}; {}/6 intervals contain zero and Holm-adjusted "
        "$p_H<.05$ for {}. These hosted results extend the controlled sanity test "
        "across model families but do not establish real-world demographic fairness "
        "or measured-human-personality effects."
    ).format(
        n,
        min(id_means),
        max(id_means),
        id_zero,
        sig_text(id_sig),
        min(p_means),
        max(p_means),
        p_zero,
        sig_text(p_sig),
    )


def _condition_group(condition_id: str) -> str | None:
    if condition_id == "C0":
        return "pref"
    if condition_id == "C1":
        return "identity"
    if condition_id.startswith("C2:"):
        return "cf_identity"
    if condition_id == "C3":
        return "true_ocean"
    if condition_id == "C4":
        return "shuffled_ocean"
    return None


def _condition_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    # Collapse the two C2 alternatives within user before averaging across users.
    by_user: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        family = str(row["model_family"])
        group = _condition_group(str(row["condition_id"]))
        if group is None:
            continue
        by_user[(family, str(row["user_id"]), group)].append(float(row["ndcg"]))

    per_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (family, _user, group), values in by_user.items():
        per_family[family][group].append(sum(values) / len(values))

    required = ("pref", "identity", "cf_identity", "true_ocean", "shuffled_ocean")
    output: dict[str, dict[str, float]] = {}
    for family in FAMILY_ORDER:
        output[family] = {}
        for group in required:
            values = per_family[family][group]
            if not values:
                raise ValueError(f"missing hosted user-condition values for {family}/{group}")
            output[family][group] = sum(values) / len(values)
    return output


def render_figure(user_condition: list[dict[str, Any]], output: Path) -> None:
    means = _condition_means(user_condition)
    groups = ("pref", "identity", "cf_identity", "true_ocean", "shuffled_ocean")
    labels = ("pref", "ID", "cf-ID", "true P", "shuf P")
    angles = [2.0 * math.pi * i / len(groups) for i in range(len(groups))]
    closed_angles = angles + angles[:1]
    global_max = max(means[f][g] for f in FAMILY_ORDER for g in groups)
    radial_max = min(1.0, max(0.4, math.ceil((global_max + 0.03) * 10.0) / 10.0))

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": "#172033",
        "axes.edgecolor": "#94A3B8",
    })
    fig, axes = plt.subplots(
        2, 3, figsize=(6.75, 4.45), subplot_kw={"projection": "polar"}
    )
    for ax, family in zip(axes.ravel(), FAMILY_ORDER, strict=True):
        values = [means[family][group] for group in groups]
        closed = values + values[:1]
        color = FAMILY_COLORS[family]
        ax.set_theta_offset(math.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.plot(closed_angles, closed, linewidth=1.4, color=color)
        ax.fill(closed_angles, closed, alpha=0.22, color=color)
        ax.scatter(angles, values, s=10, color=color, zorder=3)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=6.4)
        ax.set_ylim(0.0, radial_max)
        ticks = [radial_max * 0.5, radial_max]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{ticks[0]:.2f}", f"{ticks[1]:.2f}"], fontsize=5.6)
        ax.grid(linewidth=0.45, alpha=0.35)
        ax.spines["polar"].set_color("#CBD5E1")
        ax.set_title(
            f"{FAMILY_LABELS[family]}\n({MODEL_IDS[family]})",
            fontsize=7.2,
            fontweight="bold",
            pad=8,
        )

    fig.text(
        0.5,
        0.012,
        "Raw mean nDCG@10 on a shared radial scale; cf-ID averages the two identity alternatives within user.",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#64748B",
    )
    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.995), h_pad=0.45, w_pad=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-jsonl", required=True)
    parser.add_argument("--user-condition-jsonl", required=True)
    parser.add_argument(
        "--table",
        default="paper/generated/fairsynth_hosted_table.tex",
    )
    parser.add_argument(
        "--summary",
        default="paper/generated/fairsynth_hosted_summary.tex",
    )
    parser.add_argument(
        "--figure",
        default="paper/figures/fairsynth_hosted_profiles.pdf",
    )
    args = parser.parse_args()

    inference = _read_jsonl(Path(args.inference_jsonl))
    user_condition = _read_jsonl(Path(args.user_condition_jsonl))

    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(render_table(inference), encoding="utf-8")

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(render_summary(inference) + "\n", encoding="utf-8")

    render_figure(user_condition, Path(args.figure))
    print(json.dumps({
        "status": "PASS",
        "table": str(table),
        "summary": str(summary),
        "figure": args.figure,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
