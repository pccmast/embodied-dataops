"""Deterministic V2 fixture generation (D2-M0 profile manifest + D2-M1 binaries).

`make fixture-v2` generates:

* the canonical profile manifest (`fixtures/v2/manifest.json`) with config SHA-256
  for the two built-in profiles and the *planned* fixture descriptors;
* with ``--canonical-binaries`` it also writes the deterministic D2-M1 binary
  payloads under ``fixtures/v2/canonical/`` (LeRobot v3 6D/7D/dual-camera
  directories and MCAP ROS2 profile v2 containers, see
  :mod:`fixtures.v2.canonical_payloads`) and records their SHA-256 / status in
  the manifest.

It never downloads anything and is reproducible offline (seed 20261029).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_dataops.domain.profiles import RobotProfileConfig, load_profiles

SEED = 20261029
V2_FIXTURE_DIR = Path("fixtures/v2")
MANIFEST_PATH = V2_FIXTURE_DIR / "manifest.json"
CANONICAL_DIR = "canonical"

PLANNED_FIXTURES: list[dict[str, Any]] = [
    {
        "fixture_id": "FV2-01-good-6d",
        "kind": "lerobot-v3-directory",
        "profile": "arm6_joint_position_v1",
        "cameras": ["front"],
        "native_rates_hz": {"front": 10, "state": 10, "action": 10},
        "episodes": 1,
        "frames": 100,
        "expected": "canonical-episode/2.0, 6D aligned view valid, QC PASS, export round-trip",
    },
    {
        "fixture_id": "FV2-02-good-7d",
        "kind": "lerobot-v3-directory",
        "profile": "arm7_joint_position_v1",
        "cameras": ["front"],
        "native_rates_hz": {"front": 10, "state": 10, "action": 10},
        "episodes": 1,
        "frames": 100,
        "expected": "canonical-episode/2.0, 7D aligned view valid, QC PASS, export round-trip",
    },
    {
        "fixture_id": "FV2-03-dual-camera",
        "kind": "mcap-ros2-profile-v2",
        "profile": "arm6_joint_position_v1",
        "cameras": ["front", "wrist"],
        "native_rates_hz": {"front": 30, "wrist": 15, "state": 100, "action": 50},
        "episodes": 1,
        "duration_s": 10,
        "expected": "native camera indices retained, aligned 10Hz multi-camera view, preview not training truth",
    },
    {
        "fixture_id": "FV2-04-semantic-qc",
        "kind": "synthetic-semantics",
        "profile": "arm6_joint_position_v1",
        "cases": [
            "joint_position_out_of_range",
            "joint_velocity_spike",
            "clock_offset_unverified",
            "action_state_lag",
            "episode_incomplete",
        ],
        "expected": "exact qc-policy/2.0 rule codes/severities with evidence",
    },
    {
        "fixture_id": "FV2-05-pointcloud",
        "kind": "pointcloud2",
        "cases": [
            "little_endian_organized",
            "big_endian",
            "nonzero_field_offset_padding",
            "truncated_data",
            "missing_xyz",
            "over_one_million_points",
        ],
        "expected": "bounded parse, POINTCLOUD_* codes, summary statistics, no OOM",
    },
]


def _payload_descriptors(target: Path) -> list[dict[str, Any]]:
    """Describe the D2-M1 canonical binaries (present/missing + recorded SHA-256)."""
    from fixtures.v2.canonical_payloads import (
        LEROBOT_FIXTURES,
        MCAP_FIXTURES,
        _sha256_file,
        directory_digest,
        directory_size,
    )
    from fixtures.v2.canonical_payloads import (
        SEED as PAYLOAD_SEED,
    )

    canonical_dir = target / CANONICAL_DIR
    entries: list[dict[str, Any]] = []
    for spec in LEROBOT_FIXTURES:
        path = canonical_dir / f"{spec['fixture_id']}.lerobot"
        entry = {key: value for key, value in spec.items()}
        entry.update(
            {
                "path": f"{CANONICAL_DIR}/{path.name}",
                "digest_kind": "directory",
                "data_origin": "synthetic",
                "seed": PAYLOAD_SEED,
            }
        )
        if path.is_dir():
            entry.update(
                {
                    "sha256": directory_digest(path),
                    "byte_size": directory_size(path),
                    "status": "present",
                }
            )
        else:
            entry.update({"sha256": None, "byte_size": 0, "status": "missing"})
        entries.append(entry)
    for spec in MCAP_FIXTURES:
        path = canonical_dir / f"{spec['fixture_id']}.mcap"
        entry = {key: value for key, value in spec.items()}
        entry.update(
            {
                "path": f"{CANONICAL_DIR}/{path.name}",
                "digest_kind": "file",
                "data_origin": "synthetic",
                "seed": PAYLOAD_SEED,
                "topic_sidecar": f"{CANONICAL_DIR}/{path.with_suffix('.topics.json').name}",
            }
        )
        if path.is_file():
            entry.update(
                {
                    "sha256": _sha256_file(path),
                    "byte_size": path.stat().st_size,
                    "status": "present",
                }
            )
        else:
            entry.update({"sha256": None, "byte_size": 0, "status": "missing"})
        entries.append(entry)
    return entries


def generate_v2_fixtures(
    output_dir: Path | str = V2_FIXTURE_DIR,
    *,
    profile_file: Path | str | None = None,
    include_binaries: bool = False,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if include_binaries:
        from fixtures.v2.canonical_payloads import generate_canonical_binaries

        generate_canonical_binaries(target, profile_file=profile_file)
    profiles: list[RobotProfileConfig] = load_profiles(profile_file)
    payload_descriptors = _payload_descriptors(target)
    generated = sum(1 for entry in payload_descriptors if entry["status"] == "present")
    manifest = {
        "schema_version": "fixtures-v2-manifest/1.0",
        "seed": SEED,
        "data_origin": "synthetic",
        "status": (
            f"D2-M1 canonical binaries: {generated}/{len(payload_descriptors)} present "
            "(regenerate offline with --canonical-binaries); semantic-QC/pointcloud payloads "
            "are owned by D2-M2/M5"
        ),
        "canonical_dir": CANONICAL_DIR,
        "canonical_payloads": payload_descriptors,
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "profile_revision": profile.profile_revision,
                "authority": profile.authority,
                "state_dim": profile.state_dim,
                "action_dim": profile.action_dim,
                "joint_names": profile.joint_names,
                "config_sha256": profile.config_sha256(),
            }
            for profile in profiles
        ],
        "fixtures": PLANNED_FIXTURES,
    }
    body = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    manifest_path = target / "manifest.json"
    manifest_path.write_text(body, encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(V2_FIXTURE_DIR))
    parser.add_argument("--profile-file", default=None)
    parser.add_argument(
        "--canonical-binaries",
        action="store_true",
        help="(re)generate the D2-M1 deterministic LeRobot/MCAP binary payloads",
    )
    args = parser.parse_args(argv)
    manifest = generate_v2_fixtures(
        args.output_dir,
        profile_file=args.profile_file,
        include_binaries=args.canonical_binaries,
    )
    present = sum(1 for entry in manifest["canonical_payloads"] if entry["status"] == "present")
    print(
        f"fixture-v2: {len(manifest['profiles'])} profiles, "
        f"{len(manifest['fixtures'])} planned fixtures, "
        f"{present}/{len(manifest['canonical_payloads'])} canonical binaries present -> "
        f"{Path(args.output_dir) / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
