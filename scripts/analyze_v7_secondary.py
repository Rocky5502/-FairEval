from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from faireval.metrics import jaccard_at_k, rbo_at_k


MODEL_LABELS = {
    "phi35_local": "Phi-3.5-mini",
    "qwen25_local": "Qwen2.5-7B",
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
    return rows


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _condition_group(condition_id: str) -> str:
    if condition_id == "C0":
        return "preference_only"
    if condition_id == "C1":
        return "observed_identity"
    if condition_id.startswith("C2:"):
        return "counterfactual_identity"
    if condition_id == "C3":
        return "true_ocean"
    if condition_id == "C4":
        return "shuffled_ocean"
    raise ValueError(f"unexpected FairSynth V7 condition_id={condition_id!r}")


def _collapse_counterfactuals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce one user/model row per descriptive condition family.

    Multiple C2 identity alternatives are averaged within user/model first so
    they are never treated as independent user observations.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row["model_family"])
        user = str(row["user_id"])
        group = _condition_group(str(row["condition_id"]))
        grouped[(family, user, group)].append(row)

    out: list[dict[str, Any]] = []
    for (family, user, group), members in grouped.items():
        out.append({
            "model_family": family,
            "user_id": user,
            "condition_group": group,
            "n_source_conditions": len(members),
            "ndcg": float(np.mean([float(x["ndcg"]) for x in members])),
            "recall": float(np.mean([float(x["recall"]) for x in members])),
            "mrr": float(np.mean([float(x["mrr"]) for x in members])),
            "invalid_rate": float(np.mean([float(x["invalid_rate"]) for x in members])),
        })
    return out


def _bootstrap_mean(values: list[float], *, seed: int, samples: int) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("cannot summarize empty values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(samples, x.size))
    means = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _summarize_conditions(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_family"]), str(row["condition_group"]))].append(row)

    output: list[dict[str, Any]] = []
    for model in ("phi35_local", "qwen25_local"):
        for pos, condition in enumerate(CONDITION_ORDER):
            members = grouped[(model, condition)]
            if not members:
                raise ValueError(f"missing condition summary rows for {model}/{condition}")
            n_users = len({str(x["user_id"]) for x in members})
            row: dict[str, Any] = {
                "model_family": model,
                "condition_group": condition,
                "n_users": n_users,
            }
            for metric_index, metric in enumerate(("ndcg", "recall", "invalid_rate")):
                stats = _bootstrap_mean(
                    [float(x[metric]) for x in members],
                    seed=2027 + pos * 101 + metric_index * 17 + (0 if model == "phi35_local" else 1009),
                    samples=bootstrap_samples,
                )
                for key, value in stats.items():
                    row[f"{metric}_{key}"] = value
            output.append(row)
    return output


def _pair_distribution(rows: list[dict[str, Any]], *, contrast: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in ("phi35_local", "qwen25_local"):
        values = [
            float(row["delta_ndcg"])
            for row in rows
            if str(row["model_family"]) == model
        ]
        if not values:
            raise ValueError(f"no paired deltas for {contrast}/{model}")
        x = np.asarray(values, dtype=float)
        output.append({
            "contrast": contrast,
            "model_family": model,
            "n_users": int(x.size),
            "mean_delta_ndcg": float(x.mean()),
            "median_delta_ndcg": float(np.median(x)),
            "q1_delta_ndcg": float(np.quantile(x, 0.25)),
            "q3_delta_ndcg": float(np.quantile(x, 0.75)),
            "positive_fraction": float(np.mean(x > 0.0)),
            "negative_fraction": float(np.mean(x < 0.0)),
            "zero_fraction": float(np.mean(x == 0.0)),
        })
    return output


def _repetition_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, float]]] = defaultdict(list)
    total_cells: dict[str, int] = defaultdict(int)
    for row in rows:
        model = str(row["model_family"])
        total_cells[model] += 1
        entries = row.get("ranking_by_repetition", [])
        if not isinstance(entries, list):
            continue
        valid = {
            int(entry["repetition"]): tuple(str(x) for x in entry.get("ranking", []))
            for entry in entries
            if isinstance(entry, dict) and bool(entry.get("valid", False))
        }
        if len(valid) < 2:
            continue
        reps = sorted(valid)[:2]
        left, right = valid[reps[0]], valid[reps[1]]
        k = int(row["k"])
        by_model[model].append({
            "rbo": float(rbo_at_k(left, right, k)),
            "jaccard": float(jaccard_at_k(left, right, k)),
            "exact_order": float(left == right),
        })

    output: list[dict[str, Any]] = []
    for model in ("phi35_local", "qwen25_local"):
        values = by_model[model]
        if not values:
            raise ValueError(f"no valid repeated-ranking pairs for {model}")
        output.append({
            "model_family": model,
            "user_condition_cells": int(total_cells[model]),
            "valid_repeat_pairs": len(values),
            "valid_repeat_pair_rate": float(len(values) / total_cells[model]),
            "mean_rbo_at_10": float(np.mean([x["rbo"] for x in values])),
            "mean_jaccard_at_10": float(np.mean([x["jaccard"] for x in values])),
            "exact_order_fraction": float(np.mean([x["exact_order"] for x in values])),
        })
    return output


def _render_stability_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "{} & {}/{} ({:.1f}\\%) & {:.3f} & {:.3f} & {:.1f}\\% \\\\".format(
                MODEL_LABELS[str(row["model_family"])],
                int(row["valid_repeat_pairs"]),
                int(row["user_condition_cells"]),
                100.0 * float(row["valid_repeat_pair_rate"]),
                float(row["mean_rbo_at_10"]),
                float(row["mean_jaccard_at_10"]),
                100.0 * float(row["exact_order_fraction"]),
            )
        )
    return "\n".join([
        "% AUTO-GENERATED by scripts/analyze_v7_secondary.py; DO NOT EDIT.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Descriptive two-repetition stability in the completed V7 local run. Only user-condition cells with two semantically valid rankings contribute to RBO/Jaccard; invalid generations remain counted in the valid-pair rate.}",
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


def _render_figure(
    condition_summary: list[dict[str, Any]],
    identity_pairs: list[dict[str, Any]],
    personality_pairs: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, (ax_profile, ax_delta) = plt.subplots(
        1, 2, figsize=(6.75, 3.0), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    x = np.arange(len(CONDITION_ORDER))
    for model in ("phi35_local", "qwen25_local"):
        rows = [
            next(
                row for row in condition_summary
                if row["model_family"] == model and row["condition_group"] == condition
            )
            for condition in CONDITION_ORDER
        ]
        means = np.asarray([float(row["ndcg_mean"]) for row in rows])
        lower = means - np.asarray([float(row["ndcg_ci_low"]) for row in rows])
        upper = np.asarray([float(row["ndcg_ci_high"]) for row in rows]) - means
        ax_profile.errorbar(
            x,
            means,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.25,
            capsize=2.5,
            label=MODEL_LABELS[model],
        )
    ax_profile.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax_profile.set_ylabel("Mean nDCG@10")
    ax_profile.set_title("(a) Utility across controlled contexts", loc="left", fontsize=8.5, fontweight="bold")
    ax_profile.grid(axis="y", linewidth=0.45, alpha=0.28)
    ax_profile.spines[["top", "right"]].set_visible(False)
    ax_profile.legend(frameon=False, fontsize=7.0)

    groups: list[tuple[str, list[float]]] = []
    for contrast, source in (
        ("Identity", identity_pairs),
        ("Personality", personality_pairs),
    ):
        for model in ("phi35_local", "qwen25_local"):
            values = [
                float(row["delta_ndcg"])
                for row in source
                if str(row["model_family"]) == model
            ]
            groups.append((f"{contrast}\n{MODEL_LABELS[model]}", values))

    box = ax_delta.boxplot(
        [values for _, values in groups],
        labels=[label for label, _ in groups],
        showfliers=False,
        widths=0.58,
        patch_artist=False,
    )
    del box
    ax_delta.axhline(0.0, linestyle="--", linewidth=0.9)
    ax_delta.set_ylabel("Per-user paired ΔnDCG@10")
    ax_delta.set_title("(b) Paired-effect distributions", loc="left", fontsize=8.5, fontweight="bold")
    ax_delta.grid(axis="y", linewidth=0.45, alpha=0.28)
    ax_delta.spines[["top", "right"]].set_visible(False)
    ax_delta.tick_params(axis="x", labelsize=6.4)

    fig.text(
        0.5,
        0.012,
        "Descriptive secondary analysis of the frozen 2,880-cell V7 run; no new model/API calls and no new confirmatory hypothesis family.",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 1.0), w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render descriptive no-new-call secondary analyses from the completed FairEval V7 artifacts."
    )
    parser.add_argument(
        "--user-condition",
        default="results/analysis/fairsynth-v7-lean/user_condition.jsonl",
    )
    parser.add_argument(
        "--identity-pairs",
        default="results/analysis/fairsynth-v7-lean/identity_pairs.jsonl",
    )
    parser.add_argument(
        "--personality-pairs",
        default="results/analysis/fairsynth-v7-lean/personality_pairs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/fairsynth-v7-secondary",
    )
    parser.add_argument(
        "--paper-figure",
        default="paper/figures/v7_secondary_profiles.pdf",
    )
    parser.add_argument(
        "--paper-stability-table",
        default="paper/generated/v7_repetition_stability_table.tex",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    user_condition_path = Path(args.user_condition)
    identity_path = Path(args.identity_pairs)
    personality_path = Path(args.personality_pairs)
    for path in (user_condition_path, identity_path, personality_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    user_condition = _read_jsonl(user_condition_path)
    identity_pairs = _read_jsonl(identity_path)
    personality_pairs = _read_jsonl(personality_path)

    collapsed = _collapse_counterfactuals(user_condition)
    condition_summary = _summarize_conditions(
        collapsed,
        bootstrap_samples=args.bootstrap_samples,
    )
    pair_summary = (
        _pair_distribution(identity_pairs, contrast="synthetic_identity")
        + _pair_distribution(personality_pairs, contrast="synthetic_personality")
    )
    stability_summary = _repetition_stability(user_condition)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "faireval-v7-secondary-analysis-v1",
        "scope": "descriptive_secondary_analysis_only",
        "new_model_or_api_calls": False,
        "new_confirmatory_hypothesis_family": False,
        "inputs": {
            "user_condition_sha256": _sha(user_condition_path),
            "identity_pairs_sha256": _sha(identity_path),
            "personality_pairs_sha256": _sha(personality_path),
        },
        "condition_summary": condition_summary,
        "pair_distribution_summary": pair_summary,
        "repetition_stability_summary": stability_summary,
    }
    summary_path = out / "secondary_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _render_figure(
        condition_summary,
        identity_pairs,
        personality_pairs,
        Path(args.paper_figure),
    )
    stability_table = Path(args.paper_stability_table)
    stability_table.parent.mkdir(parents=True, exist_ok=True)
    stability_table.write_text(
        _render_stability_table(stability_summary),
        encoding="utf-8",
        newline="\\n",
    )
    print(json.dumps({
        "status": "PASS",
        "summary": str(summary_path),
        "paper_figure": args.paper_figure,
        "paper_stability_table": str(stability_table),
        "new_model_or_api_calls": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
