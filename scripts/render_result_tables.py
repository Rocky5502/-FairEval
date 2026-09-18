from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MODEL_ORDER = (
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "qwen",
    "meta",
    "qwen25_local",
    "phi35_local",
)
MODEL_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "deepseek": "DeepSeek",
    "qwen": "Qwen API",
    "meta": "Llama",
    "qwen25_local": "Qwen2.5-7B local",
    "phi35_local": "Phi-3.5-mini local",
}
TRAIT_ORDER = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
FACTOR_ORDER = ("prompt", "cue", "candidate_order", "cutoff", "stochasticity")
FACTOR_LABELS = {
    "prompt": "Task wording",
    "cue": "Cue realization",
    "candidate_order": "Candidate order",
    "cutoff": "Ranking cutoff $K$",
    "stochasticity": "Repeated generation",
}
WHITEBOX_FIELDS = {
    "mean_generated_token_logprob": "Mean token log-prob.",
    "generated_token_nll": "Token NLL",
    "generated_token_perplexity": "Token perplexity",
    "mean_top1_top2_logit_margin": "Top1--top2 margin",
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def _fmt(value: object, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _fmt_p(value: object) -> str:
    if value is None:
        return "--"
    p = float(value)
    return "$<.001$" if p < 0.001 else f"{p:.3f}"


def _effect_cell(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "--"
    return "{} [{}, {}]; $p_H={}$".format(
        _fmt(row.get("mean_paired_difference")),
        _fmt(row.get("bootstrap_ci_low")),
        _fmt(row.get("bootstrap_ci_high")),
        _fmt_p(row.get("holm_adjusted_p")),
    )


def _inference_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    output: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["rq"]),
            str(row["contrast"]),
            str(row["metric"]),
            str(row["dataset"]),
            str(row["model_family"]),
            "" if row.get("attribute") is None else str(row.get("attribute")),
        )
        if key in output:
            raise ValueError(f"duplicate inference row {key!r}")
        output[key] = row
    return output


def render_rq12_main(inference_rows: Sequence[Mapping[str, Any]] | None) -> str:
    idx = {} if inference_rows is None else _inference_index(inference_rows)
    lines = [
        "% AUTO-GENERATED/CONTRACT: main RQ1+RQ2 results table.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Main RQ1--RQ2 results. Cells report paired mean effect [95\\% bootstrap CI] and Holm-adjusted paired-permutation $p_H$. RQ1 is observed minus matched demographic-counterfactual nDCG@10; RQ2 is true measured personality minus shuffled-profile nDCG@10 (PVA).}",
        "\\label{tab:main-rq12-results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.15pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Model & RQ1 MovieLens-1M & RQ1 Last.fm-1K & RQ2 Personality 2018 & RQ2 Music Master & RQ2 REASONER \\\\",
        "\\midrule",
    ]
    for family in MODEL_ORDER:
        if inference_rows is None:
            cells = ["\\tbd"] * 5
        else:
            cells = [
                _effect_cell(
                    idx.get(
                        (
                            "RQ1",
                            "observed_vs_demographic_counterfactual",
                            "ndcg",
                            dataset,
                            family,
                            "gender",
                        )
                    )
                )
                for dataset in ("movielens_1m", "lastfm_1k")
            ]
            cells.extend(
                _effect_cell(
                    idx.get(
                        (
                            "RQ2",
                            "true_vs_shuffled_personality",
                            "ndcg",
                            dataset,
                            family,
                            "",
                        )
                    )
                )
                for dataset in ("personality2018", "music_master_bfi2", "reasoner")
            )
        lines.append(f"{MODEL_LABELS[family]} & " + " & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\vspace{0.4mm}",
            "\\parbox{0.985\\textwidth}{\\scriptsize\\textit{Reading rule.} Effect direction is descriptive; RBO/Jaccard are sensitivity diagnostics rather than fairness verdicts. Dataset-level hypotheses remain separate inside the pre-registered Holm families.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_prompting_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if summary.get("schema_version") != "faireval-rq4-prompting-summary-v1":
        raise ValueError("unexpected RQ4 prompting summary schema")
    if summary.get("intervention") != "identity_irrelevance_prompting":
        raise ValueError("RQ4 prompting summary intervention mismatch")
    if summary.get("confirmatory_p_values") is not False:
        raise ValueError("RQ4 prompting table requires confirmatory_p_values=false")
    macro = summary.get("macro")
    if not isinstance(macro, Mapping):
        raise ValueError("RQ4 prompting summary lacks macro block")
    return macro


def render_rq34_main(
    rq3_rows: Sequence[Mapping[str, Any]] | None,
    rq4_artifact: Mapping[str, Any] | None,
    rq4_prompting_summary: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        "% AUTO-GENERATED/CONTRACT: main RQ3+RQ4 results table.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Main RQ3--RQ4 reliability and mitigation results. RQ3 reports within-user nDCG@10 variation under pre-registered one-factor perturbations. RQ4 compares the audit baseline, identity-irrelevance prompting, and one validation-frozen contextual-PAIR operating point.}",
        "\\label{tab:main-rq34-results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "Block & Comparison & $N$ strata/users & Utility/stability & Gap consequence & Decision \\\\",
        "\\midrule",
    ]

    if rq3_rows is None:
        for factor in FACTOR_ORDER:
            lines.append(
                f"RQ3 & {FACTOR_LABELS[factor]} & \\tbd & \\tbd & \\tbd & report all registered levels \\\\")
    else:
        for factor in FACTOR_ORDER:
            selected = [
                row
                for row in rq3_rows
                if row.get("schema_version") == "faireval-rq3-variation-summary-v1"
                and str(row.get("factor")) == factor
                and str(row.get("metric")) == "ndcg"
            ]
            if not selected:
                raise ValueError(f"RQ3 artifact contains no nDCG rows for factor={factor!r}")
            values = [float(row["mean_within_user_sd"]) for row in selected]
            users = sum(int(row["n_users"]) for row in selected)
            stability = (
                f"median SD={statistics.median(values):.3f}; "
                f"range [{min(values):.3f},{max(values):.3f}]"
            )
            lines.append(
                f"RQ3 & {FACTOR_LABELS[factor]} & {len(values)} strata / {users} user-summaries & {stability} & -- & no best-level selection \\\\")

    lines.append("\\midrule")
    if rq4_artifact is None:
        lines.extend(
            [
                "RQ4 & Unmitigated identity-conditioned baseline & \\tbd & \\tbd & \\tbd & reference \\\\",
                "RQ4 & Identity-irrelevance prompting & \\tbd & \\tbd & \\tbd & pre-registered baseline \\\\",
                "RQ4 & Contextual PAIR (global validation-frozen point) & \\tbd & \\tbd & \\tbd & $\\geq95\\%$ validation utility floor \\\\",
            ]
        )
    else:
        if rq4_artifact.get("schema_version") != "faireval-rq4-pair-artifact-v1":
            raise ValueError("unexpected RQ4 artifact schema")
        if rq4_artifact.get("selection_used_test_outcomes") is not False:
            raise ValueError("RQ4 table refuses artifact selected using test outcomes")
        if rq4_artifact.get("per_model_or_dataset_tuning") is not False:
            raise ValueError("RQ4 table refuses per-model/per-dataset PAIR tuning")
        op = rq4_artifact["operating_point"]
        test = rq4_artifact["test_summary"]
        validation = op["validation_summary"]
        lines.append(
            "RQ4 & Unmitigated identity-conditioned baseline & {} users & nDCG={} & $|\\CUG|$={} & reference \\\\".format(
                test.get("n_users", "--"),
                _fmt(test.get("baseline_identity_ndcg_mean")),
                _fmt(test.get("baseline_abs_cug_ndcg_on_available")),
            )
        )
        if rq4_prompting_summary is None:
            lines.append(
                "RQ4 & Identity-irrelevance prompting & -- & missing prompting artifact & missing prompting artifact & cannot finalize table \\\\")
        else:
            macro = _validate_prompting_summary(rq4_prompting_summary)
            lines.append(
                "RQ4 & Identity-irrelevance prompting & {} strata & nDCG={} & $|\\CUG|$={} & descriptive intervention baseline \\\\".format(
                    macro.get("n_strata", "--"),
                    _fmt(macro.get("mean_identity_ndcg")),
                    _fmt(macro.get("mean_abs_cug_ndcg")),
                )
            )
        lines.append(
            "RQ4 & Contextual PAIR $\\alpha={},\\lambda={}$ & {} users & nDCG={} (retention={}) & $|\\CUG|$={} & validation retention={} \\\\".format(
                _fmt(op.get("alpha"), 2),
                _fmt(op.get("lambda_instability"), 2),
                test.get("n_users", "--"),
                _fmt(test.get("pair_identity_ndcg_mean_system")),
                _fmt(test.get("utility_retention")),
                _fmt(test.get("pair_abs_cug_ndcg_on_available")),
                _fmt(validation.get("utility_retention")),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\vspace{0.4mm}", "\\parbox{0.985\\textwidth}{\\scriptsize\\textit{Reading rule.} RQ3 reports all registered factor levels; RQ4 reports only validation-frozen interventions. Test outcomes never choose the mitigation operating point.}", "\\end{table*}", ""])
    return "\n".join(lines)


def render_coverage(user_condition_rows: Sequence[Mapping[str, Any]] | None) -> str:
    lines = [
        "% AUTO-GENERATED/CONTRACT: compact execution coverage table.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Execution coverage and persistent-invalid output rate after user-condition aggregation. FairSynth remains separate from real-world claims.}",
        "\\label{tab:coverage-results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.0pt}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule Dataset & Users & Models & Conditions & Invalid rate \\\\",
        "\\midrule",
    ]
    if user_condition_rows is None:
        for dataset in (
            "personality2018",
            "music_master_bfi2",
            "reasoner",
            "movielens_1m",
            "lastfm_1k",
            "mind",
            "fairsynth360",
        ):
            lines.append(f"{_esc(dataset)} & \\tbd & \\tbd & \\tbd & \\tbd \\\\")
    else:
        by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in user_condition_rows:
            by_dataset[str(row["dataset"])].append(row)
        for dataset in sorted(by_dataset):
            rows = by_dataset[dataset]
            users = len({str(row["user_id"]) for row in rows})
            models = len({str(row["model_family"]) for row in rows})
            conditions = len({str(row["condition_id"]) for row in rows})
            invalid = statistics.fmean(float(row["invalid_rate"]) for row in rows)
            lines.append(
                f"{_esc(dataset)} & {users} & {models} & {conditions} & {invalid:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_trait_table(trait_rows: Sequence[Mapping[str, Any]] | None) -> str:
    lines = [
        "% AUTO-GENERATED/CONTRACT: compact RQ2 trait robustness table.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{RQ2 one-trait robustness. Each row summarizes pre-registered dataset$\\times$model hypotheses for true measured personality minus the corresponding one-trait observed-value intervention.}",
        "\\label{tab:trait-ablation-results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{5.0pt}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule Trait & Hypotheses & Median $\\Delta$nDCG & Range \\\\",
        "\\midrule",
    ]
    if trait_rows is None:
        for trait in TRAIT_ORDER:
            lines.append(f"{trait.title()} & \\tbd & \\tbd & \\tbd \\\\")
    else:
        for trait in TRAIT_ORDER:
            values = [
                float(row["mean_paired_difference"])
                for row in trait_rows
                if str(row.get("contrast")) == "true_vs_one_trait_counterfactual"
                and str(row.get("metric")) == "ndcg"
                and str(row.get("attribute")) == trait
            ]
            if not values:
                raise ValueError(f"trait inference contains no nDCG rows for {trait}")
            lines.append(
                f"{trait.title()} & {len(values)} & {statistics.median(values):.3f} & [{min(values):.3f},{max(values):.3f}] \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_fairsynth_table(rows: Sequence[Mapping[str, Any]] | None) -> str:
    lines = [
        "% AUTO-GENERATED/CONTRACT: FairSynth-360 sanity table.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{FairSynth-360 controlled sanity checks. Synthetic A/B/C identity is relevance-invariant by construction; synthetic OCEAN is not human psychometric measurement.}",
        "\\label{tab:fairsynth-results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{5.0pt}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule Contrast / stratum & Hypotheses & Median $\\Delta$nDCG & Range \\\\",
        "\\midrule",
    ]
    contrasts = (
        (
            "SYNTH-ID",
            "observed_identity_vs_mean_counterfactual_identity",
            "Identity sanity",
        ),
        (
            "SYNTH-PERSONALITY",
            "true_synthetic_ocean_vs_shuffled_profile",
            "Synthetic personality sanity",
        ),
    )
    if rows is None:
        for _, _, label in contrasts:
            lines.append(f"{label} / hosted & \\tbd & \\tbd & \\tbd \\\\")
            lines.append(f"{label} / local & \\tbd & \\tbd & \\tbd \\\\")
    else:
        for rq, contrast, label in contrasts:
            for stratum, families in (
                ("hosted", set(MODEL_ORDER[:6])),
                ("local", set(MODEL_ORDER[6:])),
            ):
                values = [
                    float(row["mean_paired_difference"])
                    for row in rows
                    if str(row.get("rq")) == rq
                    and str(row.get("contrast")) == contrast
                    and str(row.get("metric")) == "ndcg"
                    and str(row.get("model_family")) in families
                ]
                if not values:
                    raise ValueError(f"FairSynth inference lacks {label}/{stratum} nDCG rows")
                lines.append(
                    f"{label} / {stratum} & {len(values)} & {statistics.median(values):.3f} & [{min(values):.3f},{max(values):.3f}] \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_whitebox_table(rows: Sequence[Mapping[str, Any]] | None) -> str:
    lines = [
        "% AUTO-GENERATED/CONTRACT: local white-box descriptive table.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Exploratory local white-box diagnostics. Correlations are descriptive and uncalibrated; they are not confirmatory uncertainty evidence.}",
        "\\label{tab:whitebox-results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{5.0pt}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule Diagnostic & Strata & Median $\\rho$(nDCG) & Range \\\\",
        "\\midrule",
    ]
    if rows is None:
        for label in WHITEBOX_FIELDS.values():
            lines.append(f"{label} & \\tbd & \\tbd & \\tbd \\\\")
    else:
        for field, label in WHITEBOX_FIELDS.items():
            values: list[float] = []
            for row in rows:
                corr = row.get(f"{field}_spearman_ndcg")
                if isinstance(corr, Mapping) and corr.get("rho") is not None:
                    values.append(float(corr["rho"]))
            if not values:
                lines.append(f"{label} & 0 & -- & -- \\\\")
            else:
                lines.append(
                    f"{label} & {len(values)} & {statistics.median(values):.3f} & [{min(values):.3f},{max(values):.3f}] \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the two main and four compact FairEval result tables"
    )
    parser.add_argument("--contracts-only", action="store_true")
    parser.add_argument("--inference-jsonl")
    parser.add_argument("--trait-inference-jsonl")
    parser.add_argument("--rq3-summary-jsonl")
    parser.add_argument("--rq4-artifact-json")
    parser.add_argument("--rq4-prompting-summary-json")
    parser.add_argument("--fairsynth-inference-jsonl")
    parser.add_argument("--whitebox-summary-jsonl")
    parser.add_argument("--user-condition-jsonl")
    parser.add_argument("--output-dir", default="paper/result_tables")
    args = parser.parse_args()

    if not args.contracts_only:
        required = {
            "--inference-jsonl": args.inference_jsonl,
            "--trait-inference-jsonl": args.trait_inference_jsonl,
            "--rq3-summary-jsonl": args.rq3_summary_jsonl,
            "--rq4-artifact-json": args.rq4_artifact_json,
            "--rq4-prompting-summary-json": args.rq4_prompting_summary_json,
            "--fairsynth-inference-jsonl": args.fairsynth_inference_jsonl,
            "--whitebox-summary-jsonl": args.whitebox_summary_jsonl,
            "--user-condition-jsonl": args.user_condition_jsonl,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"final rendering requires all frozen artifacts; missing {missing}")

    inference = None if args.contracts_only else _read_jsonl(Path(args.inference_jsonl))
    trait = None if args.contracts_only else _read_jsonl(Path(args.trait_inference_jsonl))
    rq3 = None if args.contracts_only else _read_jsonl(Path(args.rq3_summary_jsonl))
    rq4 = None if args.contracts_only else _read_json(Path(args.rq4_artifact_json))
    rq4_prompt = (
        None
        if args.contracts_only
        else _read_json(Path(args.rq4_prompting_summary_json))
    )
    fairsynth = None if args.contracts_only else _read_jsonl(Path(args.fairsynth_inference_jsonl))
    whitebox = None if args.contracts_only else _read_jsonl(Path(args.whitebox_summary_jsonl))
    user_condition = None if args.contracts_only else _read_jsonl(Path(args.user_condition_jsonl))

    out = Path(args.output_dir)
    outputs = {
        "rq12_main_table.tex": render_rq12_main(inference),
        "rq34_main_table.tex": render_rq34_main(rq3, rq4, rq4_prompt),
        "coverage_table.tex": render_coverage(user_condition),
        "trait_ablation_table.tex": render_trait_table(trait),
        "fairsynth_table.tex": render_fairsynth_table(fairsynth),
        "whitebox_table.tex": render_whitebox_table(whitebox),
    }
    for name, text in outputs.items():
        _write(out / name, text)
    print(
        json.dumps(
            {
                "schema_version": "faireval-paper-result-tables-v2",
                "outputs": sorted(outputs),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
