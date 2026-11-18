"""Strong ETag handling for DO-A02 conditional writes (C2).

Supported forms::

    ETag: "episode-rev:7"     If-Match: "episode-rev:7"
    ETag: "task-rev:3"        If-Match: "task-rev:3"
    ETag: "evaluation-rev:2"  If-Match: "evaluation-rev:2"

Weak validators (``W/...``) are rejected: the contract requires strong ETags so a
revision comparison cannot be downgraded by a proxy.  A missing header raises
``428 PRECONDITION_REQUIRED``; a stale, malformed, weak or wrong-kind value raises
``412 PRECONDITION_FAILED`` with the safe ``current_business_revision`` detail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from embodied_dataops.errors import ApiError, ErrorCode

_REVISION_RE = re.compile(r"^(?P<kind>[a-z-]+-rev):(?P<revision>[1-9][0-9]{0,17})$")


@dataclass(frozen=True)
class Precondition:
    kind: str
    revision: int
    raw: str


def format_etag(kind: str, revision: int) -> str:
    """Return the strong ETag value including quotes (for the response header)."""
    return f'"{kind}:{int(revision)}"'


def etag_header(kind: str, revision: int) -> dict[str, str]:
    return {"ETag": format_etag(kind, revision)}


def parse_precondition(raw: str | None, *, expected_kind: str, current_revision: int) -> Precondition:
    """Validate ``If-Match`` for one resource or raise the contract error."""
    if raw is None or not str(raw).strip():
        raise ApiError(
            ErrorCode.PRECONDITION_REQUIRED,
            "If-Match header with a strong ETag is required for this operation",
            details={"expected_etag": format_etag(expected_kind, current_revision)},
        )
    value = str(raw).strip()
    if value.startswith("W/") or value.startswith("w/"):
        raise ApiError(
            ErrorCode.PRECONDITION_FAILED,
            "Weak ETags are not accepted; use the strong ETag returned by GET",
            details={
                "reason": "weak_etag_not_allowed",
                "current_business_revision": current_revision,
                "expected_etag": format_etag(expected_kind, current_revision),
            },
        )
    # If-Match may legally carry a comma separated list; one of the entries must
    # be the strong validator for this exact resource kind.
    candidates = [item.strip() for item in value.split(",") if item.strip()]
    parsed: list[Precondition] = []
    for candidate in candidates:
        if candidate == "*":
            raise ApiError(
                ErrorCode.PRECONDITION_FAILED,
                "If-Match: * is not accepted; send the explicit business revision",
                details={
                    "reason": "explicit_revision_required",
                    "current_business_revision": current_revision,
                    "expected_etag": format_etag(expected_kind, current_revision),
                },
            )
        if not (len(candidate) >= 2 and candidate[0] == '"' and candidate[-1] == '"'):
            raise ApiError(
                ErrorCode.PRECONDITION_FAILED,
                "Malformed If-Match value: a quoted strong ETag is required",
                details={
                    "reason": "malformed_etag",
                    "current_business_revision": current_revision,
                    "expected_etag": format_etag(expected_kind, current_revision),
                },
            )
        unquoted = candidate[1:-1]
        match = _REVISION_RE.match(unquoted)
        if match is None:
            raise ApiError(
                ErrorCode.PRECONDITION_FAILED,
                "Malformed If-Match value",
                details={
                    "reason": "malformed_etag",
                    "current_business_revision": current_revision,
                    "expected_etag": format_etag(expected_kind, current_revision),
                },
            )
        parsed.append(Precondition(kind=match.group("kind"), revision=int(match.group("revision")), raw=candidate))
    matching = [item for item in parsed if item.kind == expected_kind]
    if not matching:
        raise ApiError(
            ErrorCode.PRECONDITION_FAILED,
            f"If-Match must address the {expected_kind} resource",
            details={
                "reason": "etag_kind_mismatch",
                "current_business_revision": current_revision,
                "expected_etag": format_etag(expected_kind, current_revision),
                "provided_kinds": sorted({item.kind for item in parsed}),
            },
        )
    if all(item.revision != current_revision for item in matching):
        raise ApiError(
            ErrorCode.PRECONDITION_FAILED,
            "The resource has changed since the supplied If-Match revision",
            details={
                "current_business_revision": current_revision,
                "expected_etag": format_etag(expected_kind, current_revision),
                "provided_etag": value,
            },
        )
    return next(item for item in matching if item.revision == current_revision)


def parse_requested_revision(raw: str | None, *, expected_kind: str) -> int:
    """Return the business revision carried by an If-Match value (syntax only).

    Used to detect a fence for the *requested* source revision before comparing
    it with the resource's current revision: a second client that sends the same
    stale revision must get 409 OPERATION_ALREADY_APPLIED (with the original
    operation id), not a plain 412, when an effect already exists for it.
    """
    if raw is None or not str(raw).strip():
        raise ApiError(
            ErrorCode.PRECONDITION_REQUIRED,
            "If-Match header with a strong ETag is required for this operation",
        )
    value = str(raw).strip()
    if value.startswith(("W/", "w/")):
        raise ApiError(ErrorCode.PRECONDITION_FAILED, "Weak ETags are not accepted")
    for candidate in (item.strip() for item in value.split(",") if item.strip()):
        if candidate == "*":
            raise ApiError(ErrorCode.PRECONDITION_FAILED, "If-Match: * is not accepted")
        if not (len(candidate) >= 2 and candidate[0] == '"' and candidate[-1] == '"'):
            raise ApiError(ErrorCode.PRECONDITION_FAILED, "Malformed If-Match value")
        match = _REVISION_RE.match(candidate[1:-1])
        if match is None:
            raise ApiError(ErrorCode.PRECONDITION_FAILED, "Malformed If-Match value")
        if match.group("kind") == expected_kind:
            return int(match.group("revision"))
    raise ApiError(ErrorCode.PRECONDITION_FAILED, "If-Match value is not for this resource kind")
