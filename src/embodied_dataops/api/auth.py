"""API key authentication and RBAC (SPEC 9.5).

Only SHA-256 digests of API keys are stored.  Roles are explicit permission
sets, not an inheritance chain: `reviewer` can approve/publish but cannot
upload or run pipelines, and `operator` cannot publish.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.db.models import ApiKey
from embodied_dataops.domain.enums import Role
from embodied_dataops.errors import ApiError, ErrorCode

KEY_PREFIX = "dataops_"


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key(role: Role, name: str = "") -> tuple[str, ApiKey]:
    """Create a new API key.  The plaintext is returned exactly once."""
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    record = ApiKey(
        id=__import__("embodied_dataops.ids", fromlist=["new_id"]).new_id(),
        name=name or role.value,
        key_hash=hash_api_key(raw),
        role=role.value,
        enabled=True,
    )
    return raw, record


PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.VIEWER: frozenset({"read"}),
    Role.OPERATOR: frozenset(
        {
            "read",
            "import:write",
            "pipeline:write",
            "analysis:write",
            "dataset:write",
        }
    ),
    Role.REVIEWER: frozenset({"read", "review:write", "publish"}),
    Role.ADMIN: frozenset(
        {
            "read",
            "import:write",
            "pipeline:write",
            "analysis:write",
            "dataset:write",
            "review:write",
            "publish",
            "admin",
            "metrics",
        }
    ),
}


@dataclass(frozen=True)
class Principal:
    key_id: str
    role: Role
    name: str

    def has(self, permission: str) -> bool:
        return permission in PERMISSIONS[self.role]

    def require(self, permission: str) -> None:
        if not self.has(permission):
            raise ApiError(
                ErrorCode.FORBIDDEN,
                f"Role {self.role.value} lacks permission {permission}",
                details={"required_permission": permission, "role": self.role.value},
            )


def authenticate(session: Session, raw_key: str) -> Principal:
    if not raw_key:
        raise ApiError(ErrorCode.UNAUTHENTICATED, "Missing bearer token")
    digest = hash_api_key(raw_key)
    record = session.scalars(select(ApiKey).where(ApiKey.key_hash == digest)).first()
    if record is None or not record.enabled or record.revoked_at is not None:
        raise ApiError(ErrorCode.UNAUTHENTICATED, "Invalid or revoked API key")
    return Principal(key_id=record.id, role=Role(record.role), name=record.name)


def get_principal(request: Request, session: Session) -> Principal:
    cached = getattr(request.state, "principal", None)
    if cached is not None:
        return cached
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        raise ApiError(ErrorCode.UNAUTHENTICATED, "Authorization: Bearer <API_KEY> is required")
    principal = authenticate(session, header.split(" ", 1)[1].strip())
    request.state.principal = principal
    return principal


def require_permission(permission: str) -> Callable[[Request], Principal]:
    """FastAPI dependency factory enforcing one permission."""

    def dependency(request: Request) -> Principal:
        principal: Principal = request.state.principal
        principal.require(permission)
        return principal

    return dependency


def make_principal_dependency():
    """Dependency that authenticates the caller (any authenticated role)."""

    def dependency(request: Request) -> Principal:
        return request.state.principal

    return dependency
