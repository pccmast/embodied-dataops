"""Structural contract of the V2.1 data layer and DAG (no database required).

These tests do not replace an Alembic run against PostgreSQL; they pin the parts that
can be checked in-process, so a refactor cannot silently drop a table, a constraint or
a DAG edge without a red test.
"""

from __future__ import annotations

import json

import pytest
import sqlalchemy as sa

from embodied_dataops.ai.schema import MAX_ACTIONS, SCHEMA_VERSION
from embodied_dataops.db.models import Base
from embodied_dataops.domain.enums import (
    AI_STEPS_V21,
    PIPELINE_DAG,
    PIPELINE_DAG_BY_VERSION,
    PIPELINE_DAG_V21,
    STEP_POOL,
    STEP_TIMEOUT_SECONDS,
    Step,
)

pytestmark = pytest.mark.no_db

REPO_CONTRACT = "contracts/EGO_ACTION_SCHEMA_V21.json"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table_name",
    [
        "ai_clip_inputs",
        "model_inferences",
        "model_call_attempts",
        "candidate_action_annotations",
    ],
)
def test_v21_tables_exist(table_name):
    assert table_name in Base.metadata.tables


def test_ai_clip_inputs_enforces_the_eight_frame_cap():
    table = Base.metadata.tables["ai_clip_inputs"]
    checks = " ".join(str(constraint.sqltext) for constraint in table.constraints if isinstance(constraint, sa.CheckConstraint))
    assert "frame_count" in checks
    assert "8" in checks


def test_model_inference_identity_is_unique_per_clip():
    table = Base.metadata.tables["model_inferences"]
    unique_columns = {
        tuple(sorted(column.name for column in constraint.columns))
        for constraint in table.constraints
        if isinstance(constraint, sa.UniqueConstraint)
    }
    assert (
        tuple(sorted(["run_id", "clip_input_id", "provider", "model_name", "prompt_sha256"]))
        in unique_columns
    )


def test_attempt_is_unique_per_inference_and_bounded():
    table = Base.metadata.tables["model_call_attempts"]
    unique_columns = {
        tuple(sorted(column.name for column in constraint.columns))
        for constraint in table.constraints
        if isinstance(constraint, sa.UniqueConstraint)
    }
    assert ("attempt_no", "inference_id") in unique_columns
    checks = " ".join(str(c.sqltext) for c in table.constraints if isinstance(c, sa.CheckConstraint))
    assert "attempt_no" in checks and "3" in checks


def test_candidate_actions_bound_the_action_index():
    table = Base.metadata.tables["candidate_action_annotations"]
    checks = " ".join(str(c.sqltext) for c in table.constraints if isinstance(c, sa.CheckConstraint))
    assert "action_index" in checks
    assert str(MAX_ACTIONS - 1) in checks


# ---------------------------------------------------------------------------
# tasks: per-clip fan-out
# ---------------------------------------------------------------------------


def test_tasks_have_a_unit_key_and_a_widened_unique_index():
    tasks = Base.metadata.tables["tasks"]
    assert "unit_key" in tasks.columns
    assert tasks.columns["unit_key"].nullable is False

    indexes = {index.name: index for index in tasks.indexes}
    assert "ix_tasks_run_id_step_unit" in indexes
    index = indexes["ix_tasks_run_id_step_unit"]
    assert index.unique is True
    assert [column.name for column in index.columns] == ["run_id", "step", "unit_key"]
    # The V2 index name must be gone: keeping both would make fan-out impossible.
    assert "ix_tasks_run_id_step" not in indexes


def test_step_check_constraint_lists_the_four_new_steps():
    tasks = Base.metadata.tables["tasks"]
    checks = " ".join(str(c.sqltext) for c in tasks.constraints if isinstance(c, sa.CheckConstraint))
    for step in AI_STEPS_V21:
        assert step.value in checks


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


def _acyclic(dag: dict[Step, tuple[Step, ...]]) -> bool:
    temporary: set[Step] = set()
    permanent: set[Step] = set()

    def visit(node: Step) -> bool:
        if node in permanent:
            return True
        if node in temporary:
            return False
        temporary.add(node)
        for dependency in dag.get(node, ()):
            if not visit(dependency):
                return False
        temporary.discard(node)
        permanent.add(node)
        return True

    return all(visit(node) for node in dag)


def test_v2_dag_is_byte_for_byte_unchanged():
    """OV-02: a 2.1 run uses a new DAG, a V2 run keeps the old one."""
    assert PIPELINE_DAG == {
        Step.NORMALIZE_EPISODE: (),
        Step.QUALITY_CHECK: (Step.NORMALIZE_EPISODE,),
        Step.DETECT_OBJECTS: (Step.QUALITY_CHECK,),
        Step.BUILD_PREVIEW: (Step.QUALITY_CHECK,),
        Step.READY_FOR_REVIEW: (Step.DETECT_OBJECTS, Step.BUILD_PREVIEW),
    }
    assert Step.VLM_PREANNOTATE_CLIP not in PIPELINE_DAG


def test_v21_dag_is_acyclic_and_adds_the_ai_branch():
    assert _acyclic(PIPELINE_DAG_V21)
    for step in AI_STEPS_V21:
        assert step in PIPELINE_DAG_V21
    assert PIPELINE_DAG_V21[Step.EXTRACT_KEYFRAMES] == (Step.QUALITY_CHECK,)
    assert Step.VLM_PREANNOTATE_CLIP in PIPELINE_DAG_V21[Step.MERGE_CANDIDATE_ANNOTATIONS]
    assert Step.DETECT_OBJECTS in PIPELINE_DAG_V21[Step.MERGE_CANDIDATE_ANNOTATIONS]
    assert PIPELINE_DAG_V21[Step.READY_FOR_REVIEW] == (
        Step.VALIDATE_AI_OUTPUT,
        Step.BUILD_PREVIEW,
    )


def test_pipeline_version_selects_the_dag():
    assert PIPELINE_DAG_BY_VERSION["2.0.0"] is PIPELINE_DAG
    assert PIPELINE_DAG_BY_VERSION["2.1.0"] is PIPELINE_DAG_V21


@pytest.mark.parametrize("step", list(Step))
def test_every_step_has_a_pool_and_a_timeout(step):
    assert step in STEP_POOL
    assert step in STEP_TIMEOUT_SECONDS
    assert STEP_TIMEOUT_SECONDS[step] > 0


def test_vlm_tasks_run_on_the_vision_pool():
    assert STEP_POOL[Step.VLM_PREANNOTATE_CLIP] == "vision"
    assert STEP_POOL[Step.EXTRACT_KEYFRAMES] == "cpu"


# ---------------------------------------------------------------------------
# contract file agrees with the code
# ---------------------------------------------------------------------------


def test_static_ego_schema_agrees_with_the_validator():
    schema = json.loads(open(REPO_CONTRACT, encoding="utf-8").read())
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert schema["properties"]["actions"]["maxItems"] == MAX_ACTIONS
    verbs = set(schema["properties"]["actions"]["items"]["properties"]["verb"]["enum"])
    from typing import get_args

    from embodied_dataops.ai.schema import Verb

    assert verbs == set(get_args(Verb))
    required = set(schema["required"])
    assert required == {
        "schema_version",
        "clip_start_ns",
        "clip_end_ns",
        "actions",
        "insufficient_evidence",
    }
