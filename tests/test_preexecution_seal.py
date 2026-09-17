from __future__ import annotations

from scripts.build_preexecution_seal import (
    _json_digest,
    collect_spec_hashes,
    dataset_release_blockers,
)


def test_preexecution_spec_hash_set_covers_scientific_sources() -> None:
    hashes = collect_spec_hashes()
    required = {
        "configs/study_design.yaml",
        "configs/analysis.yaml",
        "configs/models.yaml",
        "configs/local_models.yaml",
        "configs/whitebox_campaign.yaml",
        "src/faireval/plan.py",
        "src/faireval/execute.py",
        "scripts/run_hosted_budgeted.py",
        "scripts/run_whitebox_family.py",
        "paper/main.tex",
        "pyproject.toml",
    }
    assert required.issubset(hashes)
    assert not any(path.startswith("paper/generated/") for path in hashes)
    assert not any(path.startswith("paper/figures/") for path in hashes)
    assert len(_json_digest(hashes)) == 64


def test_preexecution_seal_reports_pending_real_dataset_locks() -> None:
    blockers = dataset_release_blockers()
    assert set(blockers) == {
        "personality2018",
        "music_master_bfi2",
        "reasoner",
        "movielens_1m",
        "lastfm_1k",
        "mind",
    }
    for reasons in blockers.values():
        assert "release_status_not_frozen" in reasons
        assert "raw_sha256_not_frozen" in reasons
        assert "license_not_reviewed" in reasons
