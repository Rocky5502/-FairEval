from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .datasets.factory import DATASET_IDS, build_dataset_adapter
from .freeze import freeze_dataset, verify_freeze


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
    prepare.add_argument("--dataset", required=True, choices=DATASET_IDS)
    prepare.add_argument("--raw-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--users", type=int, default=20, help="pilot default; freeze before confirmatory run")
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

    card = subparsers.add_parser("dataset-card", help="print the adapter's dataset card")
    card.add_argument("--dataset", required=True, choices=DATASET_IDS)
    card.set_defaults(func=_card)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
