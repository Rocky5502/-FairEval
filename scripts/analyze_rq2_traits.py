from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from faireval.freeze import canonical_json, file_sha256
from faireval.inference import summarize_paired_estimands
from faireval.trait_analysis import build_rq2_one_trait_pairs


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze pre-registered RQ2 C5 one-trait personality robustness"
    )
    parser.add_argument(
        "--user-condition-jsonl",
        required=True,
        help="analysis/user_condition.jsonl from scripts/analyze_core.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/rq2-traits-v1",
    )
    parser.add_argument("--analysis-config", default="configs/analysis.yaml")
    args = parser.parse_args()

    source = Path(args.user_condition_jsonl)
    config_path = Path(args.analysis_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inference_cfg = config["inference"]
    permutation_cfg = inference_cfg["paired_permutation"]

    aggregated = _read_jsonl(source)
    pairs = build_rq2_one_trait_pairs(aggregated)
    if not pairs:
        raise ValueError("no C3/C5 RQ2 trait pairs found in user-condition artifact")

    utility_metrics = [config["utility"]["primary_metric"]] + list(
        config["utility"].get("additional_metrics", [])
    )
    summaries = summarize_paired_estimands(
        pairs,
        metrics=tuple(utility_metrics),
        bootstrap_samples=int(inference_cfg["bootstrap_samples"]),
        bootstrap_seed=int(inference_cfg["bootstrap_seed"]),
        permutation_exact_max_n=int(permutation_cfg["exact_max_nonzero_pairs"]),
        permutation_samples=int(permutation_cfg["monte_carlo_samples"]),
        permutation_seed=int(permutation_cfg["seed"]),
        confidence=float(inference_cfg["confidence"]),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair_path = out / "trait_pairs.jsonl"
    inference_path = out / "trait_inference.jsonl"
    _write_jsonl(pair_path, pairs)
    _write_jsonl(inference_path, summaries)

    manifest = {
        "schema_version": "faireval-rq2-trait-analysis-manifest-v1",
        "scope": "rq2_one_trait_robustness_only",
        "source_user_condition_sha256": file_sha256(source),
        "analysis_config_sha256": file_sha256(config_path),
        "row_counts": {
            "trait_pairs": len(pairs),
            "trait_inference": len(summaries),
        },
        "artifact_sha256": {
            "trait_pairs": file_sha256(pair_path),
            "trait_inference": file_sha256(inference_path),
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
