"""API key bootstrap/list/create helpers (SPEC 9.5)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import generate_api_key, hash_api_key
from embodied_dataops.db.models import ApiKey
from embodied_dataops.domain.enums import Role

BOOTSTRAP_KEY_FILE = Path("var/bootstrap/api_keys.json")
PRINTED_ROLES = (Role.VIEWER, Role.OPERATOR, Role.REVIEWER, Role.ADMIN)
#: V2 two-person approval (SPEC §4.2, DO2-16) needs at least two reviewer keys on
#: a single-machine bootstrap.  The primary reviewer stays ``keys["reviewer"]``;
#: the second one is exposed under the separate ``reviewer_2`` file key.
SECOND_REVIEWER_NAME = "bootstrap-reviewer-2"
SECOND_REVIEWER_FILE_KEY = "reviewer_2"


def _write_private(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)


def ensure_bootstrap_keys(session: Session, *, quiet: bool = False) -> dict[str, str]:
    """Create any missing role keys, print plaintext exactly once, return them.

    The terminal output is the authoritative one-time disclosure; the local
    ``var/bootstrap/api_keys.json`` copy exists only so the one-shot seed/demo
    scripts and the Streamlit dev server can authenticate.  It is gitignored and
    is not part of the delivered configuration.
    """
    created: dict[str, str] = {}
    existing_roles = set(session.scalars(select(ApiKey.role).where(ApiKey.enabled.is_(True))))
    for role in PRINTED_ROLES:
        if role.value in existing_roles:
            continue
        raw, record = generate_api_key(role, name=f"bootstrap-{role.value}")
        session.add(record)
        created[role.value] = raw
    # Always guarantee a second, distinct reviewer identity for two-person
    # annotation approval; the primary reviewer keeps the ``reviewer`` key name.
    reviewer_names = set(
        session.scalars(
            select(ApiKey.name).where(
                ApiKey.role == Role.REVIEWER.value, ApiKey.enabled.is_(True)
            )
        )
    )
    if SECOND_REVIEWER_NAME not in reviewer_names:
        raw, record = generate_api_key(Role.REVIEWER, name=SECOND_REVIEWER_NAME)
        session.add(record)
        created[SECOND_REVIEWER_FILE_KEY] = raw
    session.flush()
    if created:
        _write_private(
            BOOTSTRAP_KEY_FILE,
            {
                "note": "local development keys - do not commit, do not use in production",
                "created_at": datetime.now(UTC).isoformat(),
                "keys": created,
            },
        )
        if not quiet:
            print("=" * 78)
            print("Initial API keys (shown once; only their SHA-256 is stored in PostgreSQL)")
            for role, raw in created.items():
                print(f"  {role:<9} {raw}")
            print(f"Local dev copy (gitignored): {BOOTSTRAP_KEY_FILE}")
            print("=" * 78)
    return created


def load_dev_keys() -> dict[str, str]:
    if not BOOTSTRAP_KEY_FILE.is_file():
        return {}
    payload = json.loads(BOOTSTRAP_KEY_FILE.read_text(encoding="utf-8"))
    return dict(payload.get("keys") or {})


def list_keys(session: Session) -> list[dict]:
    rows = list(session.scalars(select(ApiKey).order_by(ApiKey.created_at)))
    return [
        {
            "id": row.id,
            "name": row.name,
            "role": row.role,
            "enabled": row.enabled,
            "key_hash_prefix": row.key_hash[:12] + "…",
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
        }
        for row in rows
    ]


def create_key(session: Session, role: str, name: str = "") -> str:
    role_enum = Role(role)
    raw, record = generate_api_key(role_enum, name=name or f"cli-{role}")
    session.add(record)
    session.commit()
    print(f"New {role} API key (shown once): {raw}")
    return raw


def key_digest(raw: str) -> str:
    return hash_api_key(raw)
