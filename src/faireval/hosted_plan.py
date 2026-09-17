from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .freeze import file_sha256, load_frozen_instances
from .local_plan import plan_fairsynth_conditions
from .plan import expand_core_run_cells, load_model_panel
from .schema import UserInstance


FAIRSYNTH_GROUPS = ("A", "B", "C")


def _score(instance: UserInstance, *, seed: int, label: str) -> str:
    return hashlib.sha256(
        f"{seed}|{label}|{instance.dataset}|{instance.user_id}".encode("utf-8")
    ).hexdigest()


def balanced_fairsynth_subset(
    instances: Sequence[UserInstance],
    *,
    users: int,
    seed: int,
) -> list[UserInstance]:
    """Select a deterministic A/B/C-balanced FairSynth subset.

    Hosted FairSynth is an auxiliary controlled sanity layer. Keeping exact
    balance across the meaningless identity groups prevents a budget-sized
    subset from accidentally introducing identity-frequency imbalance.
    """
    if users <= 0 or users % len(FAIRSYNTH_GROUPS) != 0:
        raise ValueError("users must be a positive multiple of 3 for balanced FairSynth selection")

    grouped: dict[str, list[UserInstance]] = defaultdict(list)
    for instance in instances:
        group = str(instance.demographics.get("synthetic_identity_group", ""))
        if group not in FAIRSYNTH_GROUPS:
            raise ValueError(
                f"FairSynth user {instance.user_id!r} has invalid synthetic_identity_group={group!r}"
            )
        grouped[group].append(instance)

    per_group = users // len(FAIRSYNTH_GROUPS)
    selected: list[UserInstance] = []
    for group in FAIRSYNTH_GROUPS:
        rows = sorted(
            grouped[group],
            key=lambda item: _score(item, seed=seed, label=f"hosted_fairsynth_{group}"),
        )
        if len(rows) < per_group:
            raise ValueError(
                f"FairSynth group {group} has only {len(rows)} users; need {per_group}"
            )
        selected.extend(rows[:per_group])

    # Mix groups deterministically so a budget-limited prefix does not consist
    # of one identity group. Full selection remains exactly balanced.
    return sorted(
        selected,
        key=lambda item: _score(item, seed=seed, label="hosted_fairsynth_interleave"),
    )


def compile_hosted_fairsynth_plan(
    *,
    freeze_root: Path,
    models_yaml: Path,
    users: int = 120,
    repetitions: int = 3,
    seed: int = 1729,
    k: int = 10,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_output_tokens: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compile a hosted-only FairSynth plan without any local-model dependency."""
    synth_dir = freeze_root / "fairsynth360"
    all_instances = load_frozen_instances(synth_dir)
    selected = balanced_fairsynth_subset(all_instances, users=users, seed=seed)
    conditions = plan_fairsynth_conditions(selected, seed=seed)
    model_panel = load_model_panel(models_yaml)
    cells = expand_core_run_cells(
        conditions,
        model_panel=model_panel,
        repetitions=repetitions,
        k=k,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )

    counts = Counter(
        str(instance.demographics["synthetic_identity_group"]) for instance in selected
    )
    manifest: dict[str, Any] = {
        "schema_version": "faireval-hosted-fairsynth-plan-v1",
        "seed": int(seed),
        "scope": "hosted_auxiliary_controlled_sanity",
        "dataset": "fairsynth360",
        "freeze_manifest_sha256": file_sha256(synth_dir / "manifest.json"),
        "instances_available": len(all_instances),
        "instances_selected": len(selected),
        "identity_group_counts": dict(sorted(counts.items())),
        "balanced_identity_subset": len(set(counts.values())) == 1,
        "planned_conditions": len(conditions),
        "planned_api_cells": len(cells),
        "model_families": [row["family"] for row in model_panel],
        "models_yaml_sha256": file_sha256(models_yaml),
        "repetitions": int(repetitions),
        "k": int(k),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
        "reporting_policy": (
            "FairSynth is separately reported and never pooled with real-world "
            "RQ1/RQ2 demographic or measured-psychometric claims."
        ),
    }
    return cells, manifest
