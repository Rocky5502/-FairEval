from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "phi35_local": "Phi-3.5-mini",
    "qwen25_local": "Qwen2.5-7B",
}


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.2f}\\%"


def render(report: dict) -> str:
    families = report["families"]
    phi = families["phi35_local"]
    qwen = families["qwen25_local"]
    phi_norm = int(report["family_integrity_audits"]["phi35_local"]["rows_with_deterministic_normalization"])
    qwen_norm = int(report["family_integrity_audits"]["qwen25_local"]["rows_with_deterministic_normalization"])
    qerr = qwen.get("semantic_error_counts", {})
    causes = (
        f"{int(qerr.get('duplicate_item_ids', 0))} duplicate; "
        f"{int(qerr.get('wrong_k', 0))} wrong-$K$; "
        f"{int(qerr.get('invalid_json_syntax', 0))} invalid JSON; "
        f"{int(qerr.get('nonlist_ranked_item_ids', 0))} non-list"
    )
    rows = [
        "Phi-3.5-mini & {valid} ({vp}) & {strict} ({sp}) & {norm} & -- \\\\".format(
            valid=phi["semantic_valid_count"], vp=_pct(phi["semantic_valid_count"], phi["rows"]),
            strict=phi["strict_format_valid_count"], sp=_pct(phi["strict_format_valid_count"], phi["rows"]), norm=phi_norm,
        ),
        "Qwen2.5-7B & {valid} ({vp}) & {strict} ({sp}) & {norm} & {causes} \\\\".format(
            valid=qwen["semantic_valid_count"], vp=_pct(qwen["semantic_valid_count"], qwen["rows"]),
            strict=qwen["strict_format_valid_count"], sp=_pct(qwen["strict_format_valid_count"], qwen["rows"]), norm=qwen_norm, causes=causes,
        ),
    ]
    return "\n".join([
        "% AUTO-GENERATED from audited V7 main-run integrity report; DO NOT EDIT BY HAND.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Output reliability over the complete V7 local matrix (1,440 cells/model). Normalization removes only permitted envelopes; no generative repair is used.}",
        "\\label{tab:v7-output-quality}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Model & Semantic valid & Strict JSON & Normalized & Persistent invalid causes \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output", default="paper/generated/v7_execution_quality_table.tex")
    args = parser.parse_args()
    report = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("analysis_allowed") is not True:
        raise ValueError("V7 main-run audit must pass before paper rendering")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(report), encoding="utf-8", newline="\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
