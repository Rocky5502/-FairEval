from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .analysis import _pair_payload, _pairing_key
from .schema import OCEAN_KEYS


def build_rq2_one_trait_pairs(
    aggregated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair true measured personality (C3) with each frozen C5 trait intervention.

    C5 is a robustness analysis, not the primary RQ2 estimand. Each trait is kept
    as a separate named hypothesis through the ``attribute`` field so inference
    never treats the five interventions from one user as five independent users.
    """
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in aggregated_rows:
        condition_id = str(row.get("condition_id", ""))
        if condition_id == "C3" or condition_id.startswith("C5:"):
            grouped[_pairing_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for rows in grouped.values():
        true_rows = [row for row in rows if str(row.get("condition_id")) == "C3"]
        if len(true_rows) > 1:
            raise ValueError("multiple C3 rows found for one RQ2 trait pairing key")
        if not true_rows:
            continue
        true = true_rows[0]

        seen_traits: set[str] = set()
        for counterfactual in rows:
            condition_id = str(counterfactual.get("condition_id", ""))
            if not condition_id.startswith("C5:"):
                continue
            intervention = counterfactual.get("condition_intervention", {})
            if not isinstance(intervention, Mapping):
                raise ValueError("C5 intervention metadata is malformed")
            trait = str(intervention.get("trait", ""))
            if trait not in OCEAN_KEYS:
                raise ValueError(f"C5 intervention has unknown trait {trait!r}")
            if condition_id != f"C5:{trait}":
                raise ValueError(
                    f"C5 condition/intervention mismatch: {condition_id!r} vs trait={trait!r}"
                )
            if intervention.get("other_traits_held_fixed") is not True:
                raise ValueError("C5 must explicitly record other_traits_held_fixed=true")
            if trait in seen_traits:
                raise ValueError(f"duplicate C5 trait {trait!r} for one RQ2 pairing key")
            seen_traits.add(trait)

            output.append(
                _pair_payload(
                    true,
                    counterfactual,
                    rq="RQ2",
                    contrast="true_vs_one_trait_counterfactual",
                    extra={
                        "attribute": trait,
                        "observed_value": intervention.get("observed_value"),
                        "counterfactual_value": intervention.get("counterfactual_value"),
                        "donor_user_id": intervention.get("donor_user_id"),
                        "other_traits_held_fixed": True,
                        "robustness_only": True,
                    },
                )
            )

    return sorted(
        output,
        key=lambda row: (
            str(row["dataset"]),
            str(row["model_family"]),
            str(row["user_id"]),
            str(row["attribute"]),
        ),
    )
