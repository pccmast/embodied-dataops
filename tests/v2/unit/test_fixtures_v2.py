"""V2 fixture skeleton tests (D2-M0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures.v2.generate import generate_v2_fixtures

from embodied_dataops.domain.profiles import load_profiles

pytestmark = pytest.mark.unit


def test_fixture_v2_manifest_is_deterministic_and_matches_profiles(tmp_path: Path):
    manifest = generate_v2_fixtures(tmp_path)
    body = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert json.loads(body) == manifest
    assert manifest["seed"] == 20261029
    assert manifest["data_origin"] == "synthetic"
    assert len(manifest["profiles"]) == 2
    profiles = load_profiles()
    by_id = {profile.profile_id: profile for profile in profiles}
    for entry in manifest["profiles"]:
        profile = by_id[entry["profile_id"]]
        assert entry["config_sha256"] == profile.config_sha256()
        assert entry["state_dim"] == profile.state_dim
        assert entry["action_dim"] == profile.action_dim
    fixture_ids = {entry["fixture_id"] for entry in manifest["fixtures"]}
    assert fixture_ids == {
        "FV2-01-good-6d",
        "FV2-02-good-7d",
        "FV2-03-dual-camera",
        "FV2-04-semantic-qc",
        "FV2-05-pointcloud",
    }

    # Regeneration must be byte-identical (deterministic, offline).
    generate_v2_fixtures(tmp_path)
    assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == body


def test_repository_fixture_v2_manifest_exists():
    path = Path("fixtures/v2/manifest.json")
    if not path.is_file():
        pytest.skip("fixtures/v2/manifest.json not generated yet; run `make fixture-v2`")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["profiles"]) == 2
    assert len(payload["fixtures"]) == 5
