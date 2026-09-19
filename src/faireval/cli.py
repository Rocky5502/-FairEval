from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .datasets.factory import ALL_DATASET_IDS, build_dataset_adapter
from .execute import completed_cell_ids, execute_plan, load_and_verify_plan, pending_cells
from .freeze import canonical_json, file_sha256, freeze_dataset, verify_freeze
from .local_plan import compile_local_open_weight_plan
from .plan import compile_core_plan


def _prepare(args: argparse.Namespace) -> int:
    adapter = build_dataset_adapter(args.dataset)
    manifest = freeze_dataset(
        adapter,
        Path(args.raw_dir),
        Path(args.output_dir),
        users=args.users,
        candidate_set_size=args.candidate_set_size,
        max_history_items=args.max_history_items,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    result = verify_freeze(Path(args.output_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _card(args: argparse.Namespace) -> int:
    adapter = build_dataset_adapter(args.dataset)
    print(json.dumps(asdict(adapter.card()), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _write_plan(
    *,
    cells: list[dict[str, object]],
    manifest: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in cells),
        encoding="utf-8",
        newline="\n",
    )
    payload = {
        **manifest,
        "plan_sha256": hashlib.sha256(canonical_json(cells).encode("utf-8")).hexdigest(),
        "run_plan_file_sha256": file_sha256(plan_path),
    }
    (output_dir / "plan_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _plan_core(args: argparse.Namespace) -> int:
    freeze_root = Path(args.freeze_root)
    counterfactuals = Path(args.counterfactuals)
    models = Path(args.models)
    cells, manifest = compile_core_plan(
        freeze_root=freeze_root,
        counterfactuals_yaml=counterfactuals,
        models_yaml=models,
        seed=args.seed,
        one_trait_subset_users=args.one_trait_subset_users,
        demographic_robustness_subset_users=args.demographic_robustness_subset_users,
    )
    payload = _write_plan(
        cells=cells,
        manifest={
            **manifest,
            "source_config_sha256": {
                "counterfactuals": file_sha256(counterfactuals),
                "models": file_sha256(models),
            },
        },
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _plan_local(args: argparse.Namespace) -> int:
    cells, manifest = compile_local_open_weight_plan(
        freeze_root=Path(args.freeze_root),
        counterfactuals_yaml=Path(args.counterfactuals),
        local_models_yaml=Path(args.models),
        seed=args.seed,
        include_real_world=not args.no_real_world,
        fairsynth_users=args.fairsynth_users,
        repetitions=args.repetitions,
    )
    payload = _write_plan(
        cells=cells,
        manifest=dict(manifest),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _execute(args: argparse.Namespace) -> int:
    plan_dir = Path(args.plan_dir)
    output_jsonl = Path(args.output_jsonl)
    family_filter = set(args.family) if args.family else None

    if not args.execute:
        cells, manifest = load_and_verify_plan(plan_dir)
        completed = completed_cell_ids(output_jsonl)
        selected = pending_cells(cells, completed=completed, families=family_filter)
        if args.max_cells is not None:
            selected = selected[: args.max_cells]
        summary = {
            "mode": "dry_run",
            "plan_sha256": manifest.get("plan_sha256"),
            "planned_api_cells_total": len(cells),
            "completed_plan_cells": len(completed),
            "selected_pending_cells": len(selected),
            "family_filter": None if family_filter is None else sorted(family_filter),
            "max_cells": args.max_cells,
            "api_calls_made": 0,
            "instruction": "Re-run with --execute only after reviewing the plan and local/provider readiness.",
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    if not args.code_commit_sha:
        raise ValueError("--code-commit-sha is required with --execute")
    summary = execute_plan(
        plan_dir=plan_dir,
        freeze_root=Path(args.freeze_root),
        output_jsonl=output_jsonl,
        code_commit_sha=args.code_commit_sha,
        families=family_filter,
        max_cells=args.max_cells,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faireval",
        description="FairEval ECIR 2027 reproducible benchmark utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="materialize a deterministic dataset split and cryptographic manifest",
    )
    prepare.add_argument("--dataset", required=True, choices=ALL_DATASET_IDS)
    prepare.add_argument("--raw-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--users", type=int, default=20)
    prepare.add_argument("--candidate-set-size", type=int, default=50)
    prepare.add_argument("--max-history-items", type=int, default=20)
    prepare.add_argument("--seed", type=int, default=1729)
    prepare.set_defaults(func=_prepare)

    verify = subparsers.add_parser(
        "verify-freeze",
        help="verify instance count and SHA-256 against a frozen dataset manifest",
    )
    verify.add_argument("--output-dir", required=True)
    verify.set_defaults(func=_verify)

    card = subparsers.add_parser("dataset-card", help="print a dataset card")
    card.add_argument("--dataset", required=True, choices=ALL_DATASET_IDS)
    card.set_defaults(func=_card)

    plan = subparsers.add_parser(
        "plan-core",
        help="compile six frozen real-world datasets into immutable hosted-core run cells",
    )
    plan.add_argument("--freeze-root", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--counterfactuals", default="configs/counterfactuals.yaml")
    plan.add_argument("--models", default="configs/models.yaml")
    plan.add_argument("--seed", type=int, default=1729)
    plan.add_argument("--one-trait-subset-users", type=int, default=30)
    plan.add_argument("--demographic-robustness-subset-users", type=int, default=20)
    plan.set_defaults(func=_plan_core)

    local = subparsers.add_parser(
        "plan-local",
        help="compile the Qwen2.5/Phi-3.5 local open-weight immutable plan",
    )
    local.add_argument("--freeze-root", default="data/frozen")
    local.add_argument("--output-dir", default="results/plans/local-open-weight-v1")
    local.add_argument("--counterfactuals", default="configs/counterfactuals.yaml")
    local.add_argument("--models", default="configs/local_models.yaml")
    local.add_argument("--seed", type=int, default=1729)
    local.add_argument("--fairsynth-users", type=int, default=360)
    local.add_argument("--repetitions", type=int, default=3)
    local.add_argument(
        "--no-real-world",
        action="store_true",
        help="build a FairSynth-only local plan before third-party dataset freezes exist",
    )
    local.set_defaults(func=_plan_local)

    execute = subparsers.add_parser(
        "execute-plan",
        help="inspect or execute pending immutable run-plan cells; dry-run by default",
    )
    execute.add_argument("--plan-dir", required=True)
    execute.add_argument("--freeze-root", required=True)
    execute.add_argument("--output-jsonl", required=True)
    execute.add_argument("--family", action="append")
    execute.add_argument("--max-cells", type=int)
    execute.add_argument("--code-commit-sha")
    execute.add_argument(
        "--execute",
        action="store_true",
        help="actually run providers/models; omit for a zero-call dry run",
    )
    execute.set_defaults(func=_execute)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
