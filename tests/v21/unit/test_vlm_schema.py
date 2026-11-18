"""Strict VLM output validation tests (SPEC_V2.1 §5.2/§5.3, V21-08).

The point of this file is that **nothing is repaired**.  Each test mutates one field of
an otherwise valid response and asserts the exact error code, so a future refactor
cannot quietly start "fixing" model output.
"""

from __future__ import annotations

import json

import pytest

from embodied_dataops.ai.schema import (
    ERROR_CLIP_BOUNDS,
    ERROR_EVIDENCE_CONSISTENCY,
    ERROR_EVIDENCE_FRAME,
    ERROR_NOT_JSON,
    ERROR_SCHEMA,
    MAX_ACTIONS,
    SCHEMA_VERSION,
    insufficient_evidence_document,
    validate_candidate_document,
)

pytestmark = pytest.mark.no_db

CLIP_START = 0
CLIP_END = 8_000_000_000
#: Positions in the image list sent for this clip (0-based), not grid indices.
SENT = list(range(8))


def _good() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_start_ns": CLIP_START,
        "clip_end_ns": CLIP_END,
        "actions": [
            {
                "verb": "pick_up",
                "noun": "cup",
                "start_ns": 1_000_000_000,
                "end_ns": 3_200_000_000,
                "evidence_frame_indices": [1, 2, 3],
                "uncertainty": "medium",
            }
        ],
        "insufficient_evidence": False,
    }


def _validate(payload, sent=SENT):
    return validate_candidate_document(
        payload,
        expected_clip_start_ns=CLIP_START,
        expected_clip_end_ns=CLIP_END,
        sent_frame_indices=sent,
    )


def _codes(outcome):
    return outcome.error_codes


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


def test_valid_document_is_accepted():
    outcome = _validate(_good())
    assert outcome.is_valid
    assert outcome.action_count == 1
    assert outcome.error_codes == []


def test_insufficient_evidence_with_no_actions_is_valid():
    outcome = _validate(insufficient_evidence_document(CLIP_START, CLIP_END))
    assert outcome.is_valid
    assert outcome.action_count == 0


def test_json_string_input_is_accepted():
    outcome = _validate(json.dumps(_good()))
    assert outcome.is_valid


# ---------------------------------------------------------------------------
# static schema
# ---------------------------------------------------------------------------


def test_non_json_is_rejected():
    outcome = _validate("not json at all")
    assert not outcome.is_valid
    assert ERROR_NOT_JSON in _codes(outcome)


def test_extra_key_is_rejected():
    payload = _good()
    payload["confidence"] = 0.93
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_SCHEMA in _codes(outcome)


def test_unknown_verb_is_rejected():
    payload = _good()
    payload["actions"][0]["verb"] = "teleport"
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_SCHEMA in _codes(outcome)


def test_wrong_schema_version_is_rejected():
    payload = _good()
    payload["schema_version"] = "ego-action-candidates/2.0"
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_SCHEMA in _codes(outcome)


@pytest.mark.parametrize("noun", ["Cup", "cup tray", "cup-tray", "", "9cup"])
def test_non_normalised_noun_is_rejected(noun):
    payload = _good()
    payload["actions"][0]["noun"] = noun
    assert not _validate(payload).is_valid


def test_missing_field_is_rejected():
    payload = _good()
    del payload["actions"][0]["uncertainty"]
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_SCHEMA in _codes(outcome)


def test_duplicate_evidence_indices_are_rejected():
    payload = _good()
    payload["actions"][0]["evidence_frame_indices"] = [1, 1]
    assert not _validate(payload).is_valid


def test_end_before_start_inside_one_action_is_rejected():
    payload = _good()
    payload["actions"][0]["start_ns"] = 3_000_000_000
    payload["actions"][0]["end_ns"] = 2_000_000_000
    assert not _validate(payload).is_valid


def test_more_than_twelve_actions_is_rejected():
    payload = _good()
    template = payload["actions"][0]
    payload["actions"] = [dict(template) for _ in range(MAX_ACTIONS + 1)]
    assert not _validate(payload).is_valid


# ---------------------------------------------------------------------------
# cross-field: clip bounds
# ---------------------------------------------------------------------------


def test_clip_start_must_match_the_sent_clip():
    payload = _good()
    payload["clip_start_ns"] = 1_000_000_000
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_CLIP_BOUNDS in _codes(outcome)


def test_clip_end_must_match_the_sent_clip():
    payload = _good()
    payload["clip_end_ns"] = CLIP_END + 1
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_CLIP_BOUNDS in _codes(outcome)


def test_action_interval_cannot_escape_the_clip():
    payload = _good()
    payload["actions"][0]["end_ns"] = CLIP_END + 1_000_000_000
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_CLIP_BOUNDS not in _codes(outcome)


def test_action_starting_before_the_clip_is_rejected():
    payload = _good()
    payload["actions"][0]["start_ns"] = -1
    assert not _validate(payload).is_valid


# ---------------------------------------------------------------------------
# cross-field: evidence frames
# ---------------------------------------------------------------------------


def test_evidence_frame_that_was_never_sent_is_rejected():
    payload = _good()
    payload["actions"][0]["evidence_frame_indices"] = [7, 99]
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_EVIDENCE_FRAME in _codes(outcome)


def test_evidence_frames_must_come_from_this_clips_frame_list():
    payload = _good()
    # 8 and 9 would be the next clip's images; this clip only sent positions 0..7
    payload["actions"][0]["evidence_frame_indices"] = [8, 9]
    assert ERROR_EVIDENCE_FRAME in _codes(_validate(payload))


# ---------------------------------------------------------------------------
# cross-field: consistency
# ---------------------------------------------------------------------------


def test_insufficient_evidence_with_actions_is_contradictory():
    payload = _good()
    payload["insufficient_evidence"] = True
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_EVIDENCE_CONSISTENCY in _codes(outcome)


def test_confident_answer_with_no_actions_is_contradictory():
    payload = _good()
    payload["actions"] = []
    payload["insufficient_evidence"] = False
    outcome = _validate(payload)
    assert not outcome.is_valid
    assert ERROR_EVIDENCE_CONSISTENCY in _codes(outcome)


# ---------------------------------------------------------------------------
# no silent repair
# ---------------------------------------------------------------------------


def test_rejected_document_never_returns_a_document():
    payload = _good()
    payload["actions"][0]["evidence_frame_indices"] = [99]
    outcome = _validate(payload)
    assert outcome.document is None
    assert outcome.action_count == 0


def test_error_details_are_bounded():
    payload = _good()
    payload["actions"][0]["evidence_frame_indices"] = list(range(50, 90))
    outcome = _validate(payload)
    assert len(outcome.errors) <= 20
