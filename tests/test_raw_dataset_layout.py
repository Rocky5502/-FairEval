from pathlib import Path

from scripts.check_raw_dataset_layout import inspect_dataset


def test_required_all_contract(tmp_path: Path):
    raw = tmp_path / "movielens_1m"
    raw.mkdir()
    spec = {"required_all": ["users.dat", "movies.dat", "ratings.dat"]}
    (raw / "users.dat").write_text("x", encoding="utf-8")
    report = inspect_dataset("movielens_1m", raw, spec)
    assert report["ready_for_release_hash"] is False
    assert report["missing_required_all"] == ["movies.dat", "ratings.dat"]

    (raw / "movies.dat").write_text("x", encoding="utf-8")
    (raw / "ratings.dat").write_text("x", encoding="utf-8")
    assert inspect_dataset("movielens_1m", raw, spec)["ready_for_release_hash"] is True


def test_required_any_contract_accepts_one_named_file(tmp_path: Path):
    raw = tmp_path / "music_master_bfi2"
    raw.mkdir()
    spec = {"required_any_of": ["music_master.xlsx", "music_master.csv"]}
    (raw / "music_master.csv").write_text("x", encoding="utf-8")
    report = inspect_dataset("music_master_bfi2", raw, spec)
    assert report["ready_for_release_hash"] is True
    assert report["present_any_of"] == ["music_master.csv"]


def test_single_spreadsheet_fallback_is_explicit(tmp_path: Path):
    raw = tmp_path / "music_master_bfi2"
    raw.mkdir()
    spec = {
        "required_any_of": ["music_master.xlsx", "music_master.csv"],
        "allow_single_spreadsheet_fallback": True,
    }
    (raw / "upstream-original-name.xlsx").write_text("x", encoding="utf-8")
    report = inspect_dataset("music_master_bfi2", raw, spec)
    assert report["ready_for_release_hash"] is True
    assert report["single_spreadsheet_fallback_used"] is True
