"""Strict validation of VLM action candidates (SPEC_V2.1 §5.2, §5.3, acceptance V21-08).

The model output is **untrusted input**.  It never becomes an annotation, never
auto-approves a revision and never rewrites a robot ``action`` array.  This module is
the single gate it has to pass, and the gate is deliberately unforgiving:

* unknown keys are rejected (a model that invents ``confidence`` is a schema error,
  not a bonus field);
* ``verb`` must be in the fixed taxonomy, ``noun`` must be a lowercase snake token;
* every interval must satisfy ``clip_start <= start < end <= clip_end`` **and** the
  clip bounds must equal the clip that was actually sent;
* ``evidence_frame_indices`` must be a non-empty subset of the **positions** of the
  images actually sent for that clip (0-based, ``0..frame_count-1``).  An index the
  model never saw is proof it is not describing the evidence;
* ``insufficient_evidence`` and ``actions`` are mutually consistent: true means zero
  actions, false means at least one.

Nothing is "repaired".  A rejected document keeps a short, quotable error list so the
UI and the audit trail can show why, without storing the whole prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SCHEMA_VERSION = "ego-action-candidates/2.1"
MAX_ACTIONS = 12
MAX_ERRORS = 20

Verb = Literal["pick_up", "put_down", "move", "open", "close", "pour", "cut", "other", "uncertain"]
Uncertainty = Literal["low", "medium", "high"]

NOUN_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"

#: Stable error codes surfaced to the API and counted in the quality statistics.
ERROR_NOT_JSON = "NOT_JSON"
ERROR_SCHEMA = "SCHEMA"
ERROR_CLIP_BOUNDS = "CLIP_BOUNDS"
ERROR_ACTION_INTERVAL = "ACTION_INTERVAL"
ERROR_EVIDENCE_FRAME = "EVIDENCE_FRAME"
ERROR_EVIDENCE_CONSISTENCY = "EVIDENCE_CONSISTENCY"
ERROR_ACTION_COUNT = "ACTION_COUNT"


class ActionCandidate(BaseModel):
    """One candidate action interval."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verb: Verb
    noun: str = Field(pattern=NOUN_PATTERN)
    start_ns: int = Field(ge=0)
    end_ns: int = Field(gt=0)
    evidence_frame_indices: list[int] = Field(min_length=1)
    uncertainty: Uncertainty

    @field_validator("evidence_frame_indices")
    @classmethod
    def _unique_frames(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("evidence_frame_indices must not contain duplicates")
        if any(index < 0 for index in value):
            raise ValueError("evidence_frame_indices must be non-negative")
        return value

    @model_validator(mode="after")
    def _interval_ordered(self) -> ActionCandidate:
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")
        return self


class CandidateDocument(BaseModel):
    """The full object the model must return."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["ego-action-candidates/2.1"]
    clip_start_ns: int = Field(ge=0)
    clip_end_ns: int = Field(gt=0)
    actions: list[ActionCandidate] = Field(default_factory=list, max_length=MAX_ACTIONS)
    insufficient_evidence: bool

    @model_validator(mode="after")
    def _clip_ordered(self) -> CandidateDocument:
        if self.clip_end_ns <= self.clip_start_ns:
            raise ValueError("clip_end_ns must be greater than clip_start_ns")
        return self


@dataclass(frozen=True)
class SchemaError:
    code: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass
class ValidationOutcome:
    """Result of validating one model response."""

    document: CandidateDocument | None
    errors: list[SchemaError] = field(default_factory=list)
    action_count: int = 0

    @property
    def is_valid(self) -> bool:
        return self.document is not None and not self.errors

    @property
    def error_codes(self) -> list[str]:
        return [error.code for error in self.errors]

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "action_count": self.action_count,
            "errors": [error.as_dict() for error in self.errors],
        }


def _bounded(errors: list[SchemaError]) -> list[SchemaError]:
    return errors[:MAX_ERRORS]


def parse_candidate_document(raw: str | bytes | dict[str, Any]) -> tuple[CandidateDocument | None, list[SchemaError]]:
    """Parse JSON and validate the static schema.  Cross-field clip checks come later."""
    payload: Any
    if isinstance(raw, (str, bytes)):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            return None, [SchemaError(ERROR_NOT_JSON, f"response is not valid JSON: {exc}"[0:200])]
    else:
        payload = raw

    if not isinstance(payload, dict):
        return None, [SchemaError(ERROR_SCHEMA, "response must be a JSON object")]

    try:
        document = CandidateDocument.model_validate(payload)
    except ValidationError as exc:
        errors: list[SchemaError] = []
        for item in exc.errors()[0:MAX_ERRORS]:
            location = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
            errors.append(SchemaError(ERROR_SCHEMA, f"{location}: {item.get('msg', 'invalid')}"))
        return None, _bounded(errors)
    return document, []


def validate_candidate_document(
    raw: str | bytes | dict[str, Any],
    *,
    expected_clip_start_ns: int,
    expected_clip_end_ns: int,
    sent_frame_indices: list[int] | tuple[int, ...],
) -> ValidationOutcome:
    """Validate one model response against the clip that was actually sent.

    ``sent_frame_indices`` are the **positions** of the images uploaded for this clip
    (``0..frame_count-1``).  They are clip-local on purpose: the model sees an ordered
    list of images and can only refer to them by position.  The caller keeps the
    position -> grid-index mapping in ``ai_clip_inputs`` so the evidence stays traceable
    back to a real frame.
    """
    document, errors = parse_candidate_document(raw)
    if document is None:
        return ValidationOutcome(document=None, errors=errors)

    errors = list(errors)

    # 1. The model must echo the clip it was given, not a wider or shifted window.
    if document.clip_start_ns != int(expected_clip_start_ns):
        errors.append(
            SchemaError(
                ERROR_CLIP_BOUNDS,
                f"clip_start_ns {document.clip_start_ns} != sent clip start {int(expected_clip_start_ns)}",
            )
        )
    if document.clip_end_ns != int(expected_clip_end_ns):
        errors.append(
            SchemaError(
                ERROR_CLIP_BOUNDS,
                f"clip_end_ns {document.clip_end_ns} != sent clip end {int(expected_clip_end_ns)}",
            )
        )

    # 2. Every interval must stay inside the clip.
    allowed_frames = {int(index) for index in sent_frame_indices}
    for position, action in enumerate(document.actions):
        if not (
            document.clip_start_ns <= action.start_ns < action.end_ns <= document.clip_end_ns
        ):
            errors.append(
                SchemaError(
                    ERROR_ACTION_INTERVAL,
                    f"actions[{position}] interval [{action.start_ns}, {action.end_ns}) "
                    f"escapes clip [{document.clip_start_ns}, {document.clip_end_ns})",
                )
            )
        unseen = sorted(set(action.evidence_frame_indices) - allowed_frames)
        if unseen:
            errors.append(
                SchemaError(
                    ERROR_EVIDENCE_FRAME,
                    f"actions[{position}] cites frames that were never sent: {unseen}",
                )
            )

    # 3. "no evidence" and "here are actions" are mutually exclusive.
    if document.insufficient_evidence and document.actions:
        errors.append(
            SchemaError(
                ERROR_EVIDENCE_CONSISTENCY,
                "insufficient_evidence=true must come with an empty actions list",
            )
        )
    if not document.insufficient_evidence and not document.actions:
        errors.append(
            SchemaError(
                ERROR_EVIDENCE_CONSISTENCY,
                "insufficient_evidence=false requires at least one action",
            )
        )

    if len(document.actions) > MAX_ACTIONS:
        errors.append(
            SchemaError(ERROR_ACTION_COUNT, f"{len(document.actions)} actions exceeds {MAX_ACTIONS}")
        )

    if errors:
        return ValidationOutcome(document=None, errors=_bounded(errors), action_count=0)
    return ValidationOutcome(document=document, errors=[], action_count=len(document.actions))


def insufficient_evidence_document(clip_start_ns: int, clip_end_ns: int) -> dict[str, Any]:
    """The canonical "nothing to see" answer, used by replay fixtures and tests."""
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_start_ns": int(clip_start_ns),
        "clip_end_ns": int(clip_end_ns),
        "actions": [],
        "insufficient_evidence": True,
    }
