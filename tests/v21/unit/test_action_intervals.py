"""``ACTION_INTERVAL`` revision document tests (SPEC_V2.1 §14, V21-14).

Pure functions: validation, deterministic ordering and the shape a reviewer submits.
The CAS/DB part of the revision flow is exercised on a PostgreSQL host.
"""

from __future__ import annotations

import pytest

from embodied_dataops.annotations.action_intervals import (
    ACTION_DOCUMENT_SCHEMA,
    MAX_ACTIONS,
    action_document,
    validate_actions,
)

pytestmark = pytest.mark.no_db


def _action(**overrides):
    payload = {
        "verb": "pick_up",
        "noun": "cup",
        "start_ns": 1_000_000_000,
        "end_ns": 2_000_000_000,
        "evidence_frame_indices": [1, 2],
    }
    payload.update(overrides)
    return payload


def test_valid_actions_pass_validation():
    assert validate_actions([_action()]) == []


def test_empty_action_list_is_allowed():
    """A reviewer may legitimately approve "no action in this clip"."""
    assert validate_actions([]) == []


def test_non_positive_interval_is_rejected():
    errors = validate_actions([_action(start_ns=2, end_ns=2)])
    assert any("greater than start_ns" in error for error in errors)


def test_negative_start_is_rejected():
    assert any("start_ns must be >= 0" in error for error in validate_actions([_action(start_ns=-1)]))


def test_missing_evidence_frame_is_rejected():
    errors = validate_actions([_action(evidence_frame_indices=[])])
    assert any("evidence frame" in error for error in errors)


def test_negative_evidence_frame_is_rejected():
    errors = validate_actions([_action(evidence_frame_indices=[-1])])
    assert any("negative evidence frame" in error for error in errors)


def test_duplicate_interval_is_rejected():
    errors = validate_actions([_action(), _action()])
    assert any("duplicate interval" in error for error in errors)


def test_too_many_actions_is_rejected():
    errors = validate_actions([_action(start_ns=i, end_ns=i + 1) for i in range(MAX_ACTIONS + 1)])
    assert any("exceeds" in error for error in errors)


def test_overlapping_actions_are_allowed():
    """Two hands can act at the same time; overlap is not an error."""
    actions = [
        _action(verb="pick_up", noun="cup", start_ns=0, end_ns=10),
        _action(verb="open", noun="drawer", start_ns=5, end_ns=15),
    ]
    assert validate_actions(actions) == []


def test_document_is_ordered_and_self_describing():
    document = action_document(
        episode_id="ep_1",
        annotation_set_id="set_1",
        actions=[
            _action(start_ns=3_000_000_000, end_ns=4_000_000_000),
            _action(start_ns=1_000_000_000, end_ns=2_000_000_000, verb="move"),
        ],
        revision_no=2,
        note="fixed the start of the second action",
    )
    assert document["schema_version"] == ACTION_DOCUMENT_SCHEMA
    assert document["annotation_kind"] == "ACTION_INTERVAL"
    assert document["revision_no"] == 2
    assert document["stats"] == {"actions": 2}
    starts = [item["start_ns"] for item in document["actions"]]
    assert starts == sorted(starts)
    assert document["actions"][0]["verb"] == "move"


def test_document_does_not_mutate_the_input():
    actions = [_action()]
    snapshot = [dict(item) for item in actions]
    action_document(
        episode_id="ep_1", annotation_set_id="set_1", actions=actions, revision_no=1, note=""
    )
    assert actions == snapshot
