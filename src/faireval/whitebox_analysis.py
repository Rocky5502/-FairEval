from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from .analysis import score_run_log


LOCAL_FAMILIES = {"qwen25_local", "phi35_local"}
WHITE_BOX_FIELDS = (
    "mean_generated_token_logprob",
    "generated_token_nll",
    "generated_token_perplexity",
    "mean_top1_top2_logit_margin",
)


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
    return rows


def extract_whitebox_rows(
    output_jsonl: Path,
    *,
    plan_dir: Path,
    freeze_root: Path,
) -> list[dict[str, Any]]:
    """Join audited local run rows to deterministic recommendation utility.

    The internal score fields are retained as auxiliary diagnostics only. This
    function does not calibrate them, threshold them, or convert them into a
    confidence probability.
    """
    scored = score_run_log(
        output_jsonl,
        plan_dir=plan_dir,
        freeze_root=freeze_root,
    )
    scored_by_cell = {str(row["planned_cell_id"]): row for row in scored}
    raw_rows = _read_jsonl(output_jsonl)

    output: list[dict[str, Any]] = []
    for raw in raw_rows:
        family = str(raw.get("model_family", ""))
        if family not in LOCAL_FAMILIES:
            continue
        cell_id = str(raw.get("planned_cell_id", ""))
        score = scored_by_cell.get(cell_id)
        if score is None:
            raise ValueError(f"local row {cell_id!r} has no scored counterpart")
        metadata = raw.get("provider_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"local row {cell_id!r} provider_metadata is malformed")
        if metadata.get("local_white_box") is not True:
            raise ValueError(f"local row {cell_id!r} is missing local_white_box=true")
        if metadata.get("white_box_diagnostic_semantics") != (
            "post_processor_generation_scores_uncalibrated"
        ):
            raise ValueError(f"local row {cell_id!r} has unknown white-box semantics")

        diagnostics: dict[str, float | None] = {}
        for field in WHITE_BOX_FIELDS:
            value = metadata.get(field)
            if value is None:
                diagnostics[field] = None
            else:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"local row {cell_id!r} has non-finite {field}")
                diagnostics[field] = numeric

        output.append(
            {
                "schema_version": "faireval-local-whitebox-row-v1",
                "planned_cell_id": cell_id,
                "dataset": score["dataset"],
                "user_id": score["user_id"],
                "model_family": family,
                "requested_model_id": score["requested_model_id"],
                "condition_id": score["condition_id"],
                "repetition": score["repetition"],
                "final_valid": score["final_valid"],
                "invalid_output": score["invalid_output"],
                "ndcg": score["ndcg"],
                "recall": score["recall"],
                "mrr": score["mrr"],
                **diagnostics,
                "generated_tokens": metadata.get("generated_tokens"),
                "model_commit_hash": metadata.get("model_commit_hash"),
                "gpu_name": metadata.get("gpu_name"),
                "gpu_vram_bytes": metadata.get("gpu_vram_bytes"),
                "torch_version": metadata.get("torch_version"),
                "cuda_version": metadata.get("cuda_version"),
                "dtype": metadata.get("dtype"),
                "diagnostic_semantics": metadata[
                    "white_box_diagnostic_semantics"
                ],
                "calibrated_uncertainty": False,
            }
        )
    if not output:
        raise ValueError("run log contains no local open-weight rows")
    return output


def _safe_mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> dict[str, float | int | None]:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal length")
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return {"n": len(left), "rho": None, "p_value_descriptive": None}
    result = spearmanr(left, right)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(rho) or not math.isfinite(p_value):
        return {"n": len(left), "rho": None, "p_value_descriptive": None}
    return {"n": len(left), "rho": rho, "p_value_descriptive": p_value}


def summarize_whitebox_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Produce descriptive internal-score summaries by dataset and local model.

    Correlations are exploratory/descriptive and are not part of the confirmatory
    fairness hypothesis family. In particular, the returned p-values are never
    treated as Holm-corrected confirmatory evidence.
    """
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["dataset"]),
                str(row["model_family"]),
                str(row["requested_model_id"]),
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, family, model_id), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "schema_version": "faireval-local-whitebox-summary-v1",
            "dataset": dataset,
            "model_family": family,
            "requested_model_id": model_id,
            "n_runs": len(group),
            "n_valid": sum(bool(item["final_valid"]) for item in group),
            "invalid_rate": sum(bool(item["invalid_output"]) for item in group) / len(group),
            "analysis_scope": "exploratory_local_whitebox",
            "calibrated_uncertainty": False,
        }
        for field in WHITE_BOX_FIELDS:
            valid_values = [
                float(item[field])
                for item in group
                if bool(item["final_valid"]) and item.get(field) is not None
            ]
            invalid_values = [
                float(item[field])
                for item in group
                if bool(item["invalid_output"]) and item.get(field) is not None
            ]
            row[f"{field}_valid_mean"] = _safe_mean(valid_values)
            row[f"{field}_invalid_mean"] = _safe_mean(invalid_values)

            valid_pairs = [
                (float(item[field]), float(item["ndcg"]))
                for item in group
                if bool(item["final_valid"]) and item.get(field) is not None
            ]
            row[f"{field}_spearman_ndcg"] = _safe_spearman(
                [value for value, _ in valid_pairs],
                [utility for _, utility in valid_pairs],
            )

            invalid_pairs = [
                (float(item[field]), float(bool(item["invalid_output"])))
                for item in group
                if item.get(field) is not None
            ]
            row[f"{field}_spearman_invalidity"] = _safe_spearman(
                [value for value, _ in invalid_pairs],
                [invalid for _, invalid in invalid_pairs],
            )
        summaries.append(row)
    return summaries
