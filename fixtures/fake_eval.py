"""F09_fake_eval fixture: the deterministic simulated evaluation payload from SPEC 13.1.

The payload is the exact input used by AT-19: transparent_cup_low_light 12/30 and
opaque_cup_normal_light 27/30.  It is an *evaluation report*, not an import, so it
is generated as JSON and does not go through the upload API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fixtures.generate_synthetic import SEED, read_manifest, update_manifest

FIXTURE_ID = "F09_fake_eval"
DEFAULT_PATH = Path("fixtures/golden/F09_fake_eval.json")
EXPECTED = {
    "outcome": "exactly one CollectionOrder for transparent_cup_low_light with requested_count=36",
    "rule": "trials>=20 and success_rate<0.80 -> max(20, 2*(trials-successes))",
}

PAYLOAD = {
    "schema_version": "evaluation/1.0",
    "model_id": "demo-policy-v1",
    "results": [
        {"scenario_key": "transparent_cup_low_light", "trials": 30, "successes": 12},
        {"scenario_key": "opaque_cup_normal_light", "trials": 30, "successes": 27},
    ],
    "is_simulated": True,
}


def generate_f09(path: Path | str = DEFAULT_PATH, *, manifest_path: Path | str | None = None) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(PAYLOAD, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    target.write_text(body, encoding="utf-8")
    entry = {
        "fixture_id": FIXTURE_ID,
        "path": target.as_posix(),
        "kind": "file",
        "content_type": "application/json",
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "size_bytes": len(body.encode("utf-8")),
        "data_origin": "synthetic",
        "seed": SEED,
        "expected": dict(EXPECTED),
        "generated_by": "fixtures/fake_eval.py",
        "status": "generated",
        "notes": "simulated evaluation input for the feedback loop; not a model-training result",
    }
    from fixtures.generate_synthetic import FixtureResult, DEFAULT_MANIFEST_PATH

    result = FixtureResult(
        fixture_id=FIXTURE_ID,
        path=target,
        kind="file",
        content_type="application/json",
        sha256=entry["sha256"],
        size_bytes=entry["size_bytes"],
        status="generated",
        expected=entry["expected"],
    )
    update_manifest(Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH, [result])
    return entry


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args(argv)
    entry = generate_f09(args.path, manifest_path=args.manifest)
    print(f"generated {entry['fixture_id']} sha256={entry['sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
