from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from faireval.analysis import aggregate_repetitions, score_run_log
from faireval.freeze import canonical_json, file_sha256
from faireval.inference import summarize_paired_estimands
from faireval.rq4_prompting import (
    build_identity_irrelevance_pairs,
    summarize_prompting_pairs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen RQ4 identity-irrelevance prompting intervention"
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--freeze-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-config", default="configs/analysis.yaml")
    args = parser.parse_args()

    raw = Path(args.output_jsonl)
    plan_dir = Path(args.plan_dir)
    freeze_root = Path(args.freeze_root)
    out = Path(args.output_dir)
    config_path = Path(args.analysis_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    scored = score_run_log(raw, plan_dir=plan_dir, freeze_root=freeze_root)
    if any(str(row.get("prompt_mode")) != "identity_irrelevance" for row in scored):
        raise ValueError("RQ4 prompting run log contains a non-mitigation prompt mode")
    aggregated = aggregate_repetitions(scored)
    pairs = build_identity_irrelevance_pairs(aggregated)
    descriptive = summarize_prompting_pairs(pairs)

    inference_cfg = config["inference"]
    permutation_cfg = inference_cfg["paired_permutation"]
    inference = summarize_paired_estimands(
        pairs,
        metrics=("ndcg", "recall", "invalid_rate"),
        bootstrap_samples=int(inference_cfg["bootstrap_samples"]),
        bootstrap_seed=int(inference_cfg["bootstrap_seed"]),
        permutation_exact_max_n=int(permutation_cfg["exact_max_nonzero_pairs"]),
        permutation_samples=int(permutation_cfg["monte_carlo_samples"]),
        permutation_seed=int(permutation_cfg["seed"]),
        confidence=float(inference_cfg["confidence"]),
    )
    # RQ4 prompting inference is intervention evaluation, not part of the primary
    # RQ1 Holm family. The generated p-values remain descriptive unless a separate
    # RQ4 confirmatory family is preregistered before execution.
    for row in inference:
        row["confirmatory_p_value"] = False
        row["rq4_intervention"] = "identity_irrelevance_prompting"

    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "scored_runs": out / "scored_runs.jsonl",
        "user_condition": out / "user_condition.jsonl",
        "pairs": out / "prompting_pairs.jsonl",
        "inference": out / "prompting_inference.jsonl",
    }
    _write_jsonl(paths["scored_runs"], scored)
    _write_jsonl(paths["user_condition"], aggregated)
    _write_jsonl(paths["pairs"], pairs)
    _write_jsonl(paths["inference"], inference)
    summary_path = out / "prompting_summary.json"
    summary_path.write_text(
        json.dumps(descriptive, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "faireval-rq4-prompting-analysis-manifest-v1",
        "intervention": "identity_irrelevance_prompting",
        "input_run_sha256": file_sha256(raw),
        "plan_manifest_sha256": file_sha256(plan_dir / "plan_manifest.json"),
        "analysis_config_sha256": file_sha256(config_path),
        "confirmatory_p_values": False,
        "row_counts": {
            "scored_runs": len(scored),
            "user_condition": len(aggregated),
            "pairs": len(pairs),
            "inference": len(inference),
        },
        "artifact_sha256": {
            **{name: file_sha256(path) for name, path in paths.items()},
            "prompting_summary": file_sha256(summary_path),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
