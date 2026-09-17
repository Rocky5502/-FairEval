from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

SPEC_PREFIXES = (
    "configs/",
    "src/faireval/",
    "scripts/",
    "tests/",
    "paper/",
)
SPEC_EXACT = {
    "pyproject.toml",
    "requirements-local-gpu.txt",
    "README.md",
    ".github/workflows/ci.yml",
    ".github/workflows/local-zero-call.yml",
}
SPEC_EXCLUDES = (
    "paper/generated/",
    "paper/figures/",
    "paper/result_tables/",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _is_scientific_spec_path(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith(SPEC_EXCLUDES):
        return False
    return normalized in SPEC_EXACT or normalized.startswith(SPEC_PREFIXES)


def assert_clean_scientific_worktree() -> None:
    """Refuse to seal uncommitted files that can change the scientific study."""
    status = _git("status", "--porcelain", "--untracked-files=all")
    dirty: list[str] = []
    for raw in status.splitlines():
        if len(raw) < 4:
            continue
        path_text = raw[3:].strip()
        candidates = [part.strip() for part in path_text.split(" -> ")]
        if any(_is_scientific_spec_path(name) for name in candidates):
            dirty.append(raw)
    if dirty:
        raise RuntimeError(
            "cannot build pre-execution seal from a dirty scientific worktree; "
            "commit or intentionally revert these files first:\n" + "\n".join(dirty)
        )


def collect_spec_hashes(root: Path = ROOT) -> dict[str, str]:
    """Hash tracked files that define the executable scientific specification."""
    tracked = _git("ls-files").splitlines()
    selected: dict[str, str] = {}
    for raw in tracked:
        name = raw.replace("\\", "/")
        if _is_scientific_spec_path(name):
            path = root / name
            if path.is_file():
                selected[name] = _sha256(path)
    if not selected:
        raise RuntimeError("scientific specification hash set is empty")
    return dict(sorted(selected.items()))


def dataset_release_blockers(
    path: Path = ROOT / "configs" / "dataset_releases.yaml",
) -> dict[str, list[str]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    datasets = cfg.get("datasets") if isinstance(cfg, dict) else None
    if not isinstance(datasets, dict):
        raise ValueError("dataset release configuration is malformed")

    blockers: dict[str, list[str]] = {}
    for dataset_id, raw in datasets.items():
        if not isinstance(raw, dict):
            raise ValueError(f"dataset release row {dataset_id!r} must be a mapping")
        reasons: list[str] = []
        if raw.get("release_status") != "frozen":
            reasons.append("release_status_not_frozen")
        digest = raw.get("raw_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            reasons.append("raw_sha256_not_frozen")
        if raw.get("license_reviewed") is not True:
            reasons.append("license_not_reviewed")
        raw_path = raw.get("raw_path")
        if not isinstance(raw_path, str) or not (ROOT / raw_path).exists():
            reasons.append("local_raw_path_missing")
        if reasons:
            blockers[str(dataset_id)] = reasons
    return dict(sorted(blockers.items()))


def _run(label: str, args: list[str], command_dir: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    payload = {
        "label": label,
        "command": [sys.executable, *args],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    command_dir.mkdir(parents=True, exist_ok=True)
    (command_dir / f"{label}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pre-execution seal step {label!r} failed with code {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _markdown(seal: dict[str, Any]) -> str:
    blockers = seal["real_dataset_blockers"]
    blocker_lines = [
        f"- `{dataset}`: {', '.join(reasons)}" for dataset, reasons in blockers.items()
    ] or ["- none"]
    return "\n".join(
        [
            "# FairEval pre-execution scientific seal",
            "",
            f"- Git commit: `{seal['git_commit_sha']}`",
            f"- Scientific specification SHA-256: `{seal['scientific_spec_sha256']}`",
            f"- Tracked specification files: {seal['scientific_spec_file_count']}",
            f"- FairSynth white-box core cells: {seal['plans']['whitebox_core']['planned_cells']}",
            f"- Hosted FairSynth planned cells: {seal['plans']['hosted_fairsynth']['planned_cells']}",
            "- Hosted API generation calls made while building this seal: 0",
            "- Local model weights loaded while building this seal: no",
            "- Empirical result numbers inserted: no",
            "- Scientific worktree clean at seal creation: yes",
            "",
            "## Real-dataset blockers",
            "",
            *blocker_lines,
            "",
            "## Interpretation",
            "",
            "This artifact freezes the executable scientific specification before paid hosted or local-GPU model generation. Derived paper result tables and figures are regenerated from their sealed renderer/source contracts and are not themselves treated as executable source. It is not an empirical result artifact. Real-world execution remains blocked until the exact third-party releases are locally frozen, license-reviewed, and hashed.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a zero-call FairEval pre-execution seal: regenerate deterministic "
            "FairSynth plans/contracts, hash the tracked scientific specification, and "
            "record unresolved third-party dataset blockers."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "preexecution" / "seal-v1",
    )
    args = parser.parse_args()

    # The seal's commit SHA must describe the actual scientific source bytes.
    # Derived paper result tables/figures may be regenerated below, but prompts,
    # configs, code, tests, paper entrypoints and renderer contracts may not be dirty.
    assert_clean_scientific_worktree()

    output_dir = args.output_dir
    command_dir = output_dir / "commands"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every step below is deterministic and zero-call: no hosted generation and no
    # local model weight loading. The 360-user FairSynth freeze is regenerated first
    # so both plans are compiled from the canonical synthetic benchmark.
    steps = [
        ("build_fairsynth360", ["scripts/build_fairsynth360.py"]),
        (
            "build_whitebox_campaign",
            [
                "scripts/build_whitebox_campaign.py",
                "--output-root",
                "results/plans/whitebox-full-v1",
            ],
        ),
        (
            "plan_hosted_fairsynth",
            [
                "scripts/plan_hosted_fairsynth.py",
                "--output-dir",
                "results/plans/hosted-fairsynth-budget-v1",
            ],
        ),
        (
            "render_result_contracts",
            ["scripts/render_result_tables_lncs.py", "--contracts-only"],
        ),
        ("build_paper_figures", ["scripts/build_paper_figures.py"]),
        ("check_config_consistency", ["scripts/check_config_consistency.py"]),
        ("check_pilot_readiness", ["scripts/check_pilot_readiness.py"]),
        ("check_paper_source", ["scripts/check_paper_source.py"]),
        ("check_result_table_contracts", ["scripts/check_result_table_contracts.py"]),
        (
            "build_preresult_overleaf",
            [
                "scripts/build_overleaf_bundle.py",
                "--exclude-results",
                "--output",
                str(output_dir / "FairEval_ECIR2027_PreResult_Overleaf.zip"),
            ],
        ),
    ]
    step_results = [_run(label, command, command_dir) for label, command in steps]

    # Deterministic builders may rewrite derived table/figure products, but they
    # must not rewrite any sealed scientific source contract.
    assert_clean_scientific_worktree()

    whitebox_manifest_path = ROOT / "results/plans/whitebox-full-v1/core/plan_manifest.json"
    hosted_manifest_path = ROOT / "results/plans/hosted-fairsynth-budget-v1/plan_manifest.json"
    fairsynth_manifest_path = ROOT / "data/frozen/fairsynth360/manifest.json"
    whitebox_manifest = _load_json(whitebox_manifest_path)
    hosted_manifest = _load_json(hosted_manifest_path)
    fairsynth_manifest = _load_json(fairsynth_manifest_path)

    whitebox_cells = int(whitebox_manifest.get("planned_api_cells", -1))
    hosted_cells = int(hosted_manifest.get("planned_api_cells", -1))
    if whitebox_cells != 12960:
        raise RuntimeError(f"white-box FairSynth geometry drift: expected 12960, got {whitebox_cells}")
    if hosted_cells != 12960:
        raise RuntimeError(f"hosted FairSynth geometry drift: expected 12960, got {hosted_cells}")

    spec_hashes = collect_spec_hashes()
    blockers = dataset_release_blockers()
    overleaf_path = output_dir / "FairEval_ECIR2027_PreResult_Overleaf.zip"
    seal: dict[str, Any] = {
        "schema_version": "faireval-preexecution-seal-v1",
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "git_branch_or_detached": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "scientific_worktree_clean": True,
        "scientific_spec_sha256": _json_digest(spec_hashes),
        "scientific_spec_file_count": len(spec_hashes),
        "scientific_spec_files": spec_hashes,
        "plans": {
            "whitebox_core": {
                "manifest_path": str(whitebox_manifest_path.relative_to(ROOT)),
                "manifest_sha256": _sha256(whitebox_manifest_path),
                "plan_sha256": whitebox_manifest.get("plan_sha256"),
                "planned_cells": whitebox_cells,
                "model_families": whitebox_manifest.get("model_families"),
            },
            "hosted_fairsynth": {
                "manifest_path": str(hosted_manifest_path.relative_to(ROOT)),
                "manifest_sha256": _sha256(hosted_manifest_path),
                "plan_sha256": hosted_manifest.get("plan_sha256"),
                "planned_cells": hosted_cells,
                "model_families": hosted_manifest.get("model_families"),
            },
        },
        "fairsynth_freeze": {
            "manifest_path": str(fairsynth_manifest_path.relative_to(ROOT)),
            "manifest_sha256": _sha256(fairsynth_manifest_path),
            "dataset": fairsynth_manifest.get("dataset"),
            "instances": fairsynth_manifest.get("instance_count"),
        },
        "real_dataset_blockers": blockers,
        "real_world_execution_ready": not bool(blockers),
        "hosted_api_generation_calls_made": 0,
        "local_model_weights_loaded": False,
        "empirical_results_seen_or_inserted": False,
        "preresult_overleaf": {
            "path": str(overleaf_path.relative_to(ROOT)),
            "sha256": _sha256(overleaf_path),
        },
        "zero_call_steps": [
            {"label": row["label"], "returncode": row["returncode"]} for row in step_results
        ],
        "execution_rule": (
            "Any change to the tracked scientific specification after this seal requires "
            "regenerating the seal and using the new commit/specification digest for execution."
        ),
    }
    seal["seal_sha256_without_self"] = _json_digest(seal)

    seal_path = output_dir / "PREEXECUTION_SEAL.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "PREEXECUTION_SEAL.md").write_text(
        _markdown(seal),
        encoding="utf-8",
    )
    print(json.dumps(seal, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
