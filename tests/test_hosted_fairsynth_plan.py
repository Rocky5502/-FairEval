from __future__ import annotations

from collections import Counter
from pathlib import Path

from faireval.freeze import freeze_dataset
from faireval.datasets.fairsynth360 import FairSynth360Adapter
from faireval.hosted_plan import compile_hosted_fairsynth_plan


ROOT = Path(__file__).resolve().parents[1]


def test_hosted_fairsynth_plan_is_balanced_and_six_family(tmp_path: Path) -> None:
    freeze_root = tmp_path / "frozen"
    freeze_dataset(
        FairSynth360Adapter(),
        tmp_path / "unused",
        freeze_root / "fairsynth360",
        users=12,
        candidate_set_size=30,
        max_history_items=8,
        seed=1729,
    )

    cells, manifest = compile_hosted_fairsynth_plan(
        freeze_root=freeze_root,
        models_yaml=ROOT / "configs" / "models.yaml",
        users=6,
        repetitions=1,
        seed=1729,
    )

    # 6 users x 6 registered FairSynth conditions x 6 hosted families.
    assert len(cells) == 216
    assert manifest["planned_api_cells"] == 216
    assert manifest["balanced_identity_subset"] is True
    assert manifest["identity_group_counts"] == {"A": 2, "B": 2, "C": 2}

    families = Counter(str(row["model_family"]) for row in cells)
    assert set(families) == {"openai", "anthropic", "google", "deepseek", "qwen", "meta"}
    assert len(set(families.values())) == 1
    assert all(row.get("seed") is None for row in cells)

    model_ids = {str(row["model_id"]) for row in cells}
    assert model_ids == {
        "gpt-5.6-terra",
        "claude-sonnet-5",
        "gemini-3.8-flash",
        "deepseek-v4.1-flash",
        "qwen3.8-max",
        "llama-4-maverick",
    }
