from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


METRICS = ("ndcg", "recall", "mrr", "invalid_output")
PRIMARY_LEVEL = {
    "prompt": "field_v2_a",
    "cue": "structured_key_value",
    "candidate_order": "adapter_frozen_order",
    "cutoff": "k_10",
}


def _core_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row["dataset"]),
        str(row["user_id"]),
        str(row["model_family"]),
        str(row["requested_model_id"]),
        str(row["condition_id"]),
    )


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "invalid_output":
        return 1.0 if bool(row["invalid_output"]) else 0.0
    return float(row[metric])


def build_factor_rows(
    *,
    core_scored: Sequence[Mapping[str, Any]],
    rq3_scored: Sequence[Mapping[str, Any]],
    rq3_plan_by_cell: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create an auditable long-form RQ3 factor table.

    For prompt/cue/order/K blocks, the baseline is repetition 0 from the core run.
    Extra levels come only from the RQ3 extra plan. Stochasticity is analyzed from
    its dedicated repeated extra cells and therefore has no injected core level.
    """
    core_rep0: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in core_scored:
        if int(row["repetition"]) != 0:
            continue
        key = _core_key(row)
        if key in core_rep0:
            raise ValueError(f"duplicate core repetition-0 row for {key}")
        core_rep0[key] = row

    output: list[dict[str, Any]] = []
    baseline_added: set[tuple[str, tuple[str, ...]]] = set()
    for row in rq3_scored:
        cell_id = str(row["planned_cell_id"])
        planned = rq3_plan_by_cell.get(cell_id)
        if planned is None:
            raise ValueError(f"RQ3 scored row references unknown planned cell {cell_id}")
        factor = str(planned.get("robustness_factor", "")).strip()
        level = str(planned.get("robustness_level", "")).strip()
        if factor not in {"prompt", "cue", "candidate_order", "cutoff", "stochasticity"}:
            raise ValueError(f"unsupported RQ3 robustness factor {factor!r}")
        if not level:
            raise ValueError(f"RQ3 cell {cell_id} lacks robustness_level")

        key = _core_key(row)
        core = core_rep0.get(key)
        if factor != "stochasticity":
            if core is None:
                raise ValueError(f"RQ3 factor {factor} has no matching core baseline for {key}")
            if str(core["code_commit_sha"]) != str(row["code_commit_sha"]):
                raise ValueError(
                    "RQ3 factor comparison crosses code commits; freeze/re-run before inference: "
                    f"core={core['code_commit_sha']} extra={row['code_commit_sha']}"
                )
            baseline_key = (factor, key)
            if baseline_key not in baseline_added:
                baseline_added.add(baseline_key)
                for metric in METRICS:
                    output.append(
                        {
                            "schema_version": "faireval-rq3-factor-row-v1",
                            "factor": factor,
                            "level": PRIMARY_LEVEL[factor],
                            "is_primary_baseline": True,
                            "dataset": core["dataset"],
                            "user_id": core["user_id"],
                            "model_family": core["model_family"],
                            "requested_model_id": core["requested_model_id"],
                            "condition_id": core["condition_id"],
                            "metric": metric,
                            "value": _metric_value(core, metric),
                            "repetition": int(core["repetition"]),
                            "planned_cell_id": core["planned_cell_id"],
                            "code_commit_sha": core["code_commit_sha"],
                        }
                    )

        for metric in METRICS:
            output.append(
                {
                    "schema_version": "faireval-rq3-factor-row-v1",
                    "factor": factor,
                    "level": level,
                    "is_primary_baseline": False,
                    "dataset": row["dataset"],
                    "user_id": row["user_id"],
                    "model_family": row["model_family"],
                    "requested_model_id": row["requested_model_id"],
                    "condition_id": row["condition_id"],
                    "metric": metric,
                    "value": _metric_value(row, metric),
                    "repetition": int(row["repetition"]),
                    "planned_cell_id": row["planned_cell_id"],
                    "code_commit_sha": row["code_commit_sha"],
                }
            )

    return sorted(
        output,
        key=lambda row: (
            row["factor"],
            row["dataset"],
            row["model_family"],
            row["user_id"],
            row["condition_id"],
            row["metric"],
            row["level"],
            row["repetition"],
        ),
    )


def summarize_factor_variation(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize within-user factor variability without treating levels as users."""
    grouped_user: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["factor"]),
            str(row["dataset"]),
            str(row["model_family"]),
            str(row["requested_model_id"]),
            str(row["condition_id"]),
            str(row["user_id"]),
            str(row["metric"]),
        )
        grouped_user[key].append(row)

    user_stats: list[dict[str, Any]] = []
    for key, values in grouped_user.items():
        factor, dataset, family, model_id, condition_id, user_id, metric = key
        numeric = [float(row["value"]) for row in values]
        levels = {str(row["level"]) for row in values}
        if factor == "stochasticity":
            if len(numeric) < 2:
                continue
            variability = statistics.pstdev(numeric)
            level_count = len(levels)
            observation_count = len(numeric)
        else:
            if len(levels) < 2:
                continue
            # One observation per level is expected for factorized blocks.
            by_level: dict[str, float] = {}
            for row in values:
                level = str(row["level"])
                if level in by_level:
                    raise ValueError(
                        f"duplicate RQ3 factor level for one user: factor={factor}, level={level}, user={user_id}"
                    )
                by_level[level] = float(row["value"])
            numeric = list(by_level.values())
            variability = statistics.pstdev(numeric)
            level_count = len(by_level)
            observation_count = len(by_level)

        user_stats.append(
            {
                "factor": factor,
                "dataset": dataset,
                "model_family": family,
                "requested_model_id": model_id,
                "condition_id": condition_id,
                "user_id": user_id,
                "metric": metric,
                "within_user_sd": variability,
                "within_user_range": max(numeric) - min(numeric),
                "level_count": level_count,
                "observation_count": observation_count,
            }
        )

    grouped_summary: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in user_stats:
        key = (
            str(row["factor"]),
            str(row["dataset"]),
            str(row["model_family"]),
            str(row["requested_model_id"]),
            str(row["condition_id"]),
            str(row["metric"]),
        )
        grouped_summary[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, values in grouped_summary.items():
        factor, dataset, family, model_id, condition_id, metric = key
        sds = [float(row["within_user_sd"]) for row in values]
        ranges = [float(row["within_user_range"]) for row in values]
        summaries.append(
            {
                "schema_version": "faireval-rq3-variation-summary-v1",
                "factor": factor,
                "dataset": dataset,
                "model_family": family,
                "requested_model_id": model_id,
                "condition_id": condition_id,
                "metric": metric,
                "n_users": len(values),
                "mean_within_user_sd": statistics.fmean(sds),
                "median_within_user_sd": statistics.median(sds),
                "mean_within_user_range": statistics.fmean(ranges),
                "median_within_user_range": statistics.median(ranges),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["metric"],
            row["factor"],
            row["dataset"],
            row["model_family"],
            row["condition_id"],
        ),
    )
