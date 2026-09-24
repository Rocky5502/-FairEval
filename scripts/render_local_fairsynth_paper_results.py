from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_LABELS = {
    "phi35_local": "Phi-3.5-mini",
    "qwen25_local": "Qwen2.5-7B",
}
RQ_LABELS = {
    "SYNTH-ID": "Identity sanity",
    "SYNTH-PERSONALITY": "Synthetic personality",
}


def _read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("inference artifact is empty")
    return rows


def _f(value: object) -> str:
    return f"{float(value):.3f}"


def render(rows: list[dict[str, Any]]) -> str:
    index = {(str(r["rq"]), str(r["model_family"]), str(r["metric"])): r for r in rows}
    body = []
    for rq in ("SYNTH-ID", "SYNTH-PERSONALITY"):
        for model in ("phi35_local", "qwen25_local"):
            ndcg = index[(rq, model, "ndcg")]
            recall = index[(rq, model, "recall")]
            body.append(
                "{} & {} & {} [{}, {}] & {} & {} [{}, {}] & {} \\\\".format(
                    RQ_LABELS[rq],
                    MODEL_LABELS[model],
                    _f(ndcg["mean_paired_difference"]),
                    _f(ndcg["bootstrap_ci_low"]),
                    _f(ndcg["bootstrap_ci_high"]),
                    _f(ndcg["holm_adjusted_p"]),
                    _f(recall["mean_paired_difference"]),
                    _f(recall["bootstrap_ci_low"]),
                    _f(recall["bootstrap_ci_high"]),
                    _f(recall["holm_adjusted_p"]),
                )
            )
    return "\n".join([
        "% AUTO-GENERATED from audited V7 FairSynth inference artifact; DO NOT EDIT BY HAND.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Local FairSynth-360 controlled sanity checks ($N=120$ paired users/model). Values are paired mean differences with 95\\% bootstrap CIs and Holm-adjusted $p_H$; synthetic controls are not pooled with real-world claims.}",
        "\\label{tab:fairsynth-local-generated}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Contrast & Model & $\\Delta$nDCG@10 [95\\% CI] & $p_H$ & $\\Delta$Recall@10 [95\\% CI] & $p_H$ \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}}",
        "\\vspace{0.4mm}",
        "\\parbox{0.985\\linewidth}{\\scriptsize No local FairSynth contrast provides evidence of a reliable nonzero utility effect under the preregistered uncertainty analysis.}",
        "\\end{table}",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-jsonl", required=True)
    parser.add_argument("--output", default="paper/generated/fairsynth_local_table.tex")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(_read(Path(args.inference_jsonl))), encoding="utf-8", newline="\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
