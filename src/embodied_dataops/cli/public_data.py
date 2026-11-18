"""Explicit public-dataset download (SPEC 13.2, AT-24).

Nothing here runs during build/test by default.  The user must ask for it with
``make fetch-public-example``; the downloaded data is never committed.

The repository id/revision are pinned in ``docs/public-dataset-provenance.md``
once a real LeRobot v3 dataset has been verified in this environment.
"""

from __future__ import annotations

import json
from pathlib import Path


def fetch_public_example(repo: str | None, revision: str | None, target: str | None) -> int:
    provenance = Path("docs/public-dataset-provenance.md")
    if not repo:
        print(
            "A real public LeRobot v3 dataset must be selected explicitly.\n"
            f"See {provenance} for the pinned repo id and revision, then run:\n"
            "  make fetch-public-example  # or:\n"
            "  uv run python -m embodied_dataops.cli.main fetch-public-example "
            '--repo <owner/name> --revision <commit> --target ./public-data',
            file=__import__("sys").stderr,
        )
        return 2
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print(
            "huggingface_hub is not installed in this environment "
            "(it is part of the optional `lerobot` extra). Install it, or download "
            "the dataset manually and point the importer at the local directory.",
            file=__import__("sys").stderr,
        )
        return 3
    target_path = Path(target or "./public-data")
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"[public-data] downloading {repo}@{revision or 'main'} -> {target_path}")
    path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target_path),
        local_dir_use_symlinks=False,
    )
    metadata = {
        "repo_id": repo,
        "revision": revision,
        "local_path": str(path),
        "license": "see dataset card",
        "data_origin": "public_dataset",
    }
    (target_path / "dataops-download.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[public-data] done; record the repo revision and licence in docs/public-dataset-provenance.md")
    return 0
