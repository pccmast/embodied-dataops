"""V2.1 M2: AI clip inputs, model inferences, call attempts, candidate actions.

Revision ID: v21_002_ai_inference_and_clips
Revises: v21_001_profile_dim_provenance

Two things happen here and they are deliberately in one revision, because a half
applied AI schema would be worse than none:

1. **Per-clip fan-out becomes representable.**  ``tasks`` had
   ``UNIQUE (run_id, step)``, which assumes one task per step per run.  SPEC_V2.1 §5.1
   requires one ``VLM_PREANNOTATE_CLIP`` task per clip (up to 75).  A nullable-free
   ``unit_key`` column (``''`` for singleton steps, the ``clip_id`` for clip tasks)
   widens the key without weakening the old guarantee: two synchronous steps in the
   same run still cannot both exist.
2. **Four new tables** exactly as SPEC_V2.1 §13 specifies.  Types and constraints are
   not relaxed: ``model_inferences`` is one logical task per clip, while
   ``model_call_attempts`` is one *billed* HTTP attempt, which is how duplicate remote
   charges stay visible instead of hidden.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "v21_002_ai_inference_and_clips"
down_revision = "v21_001_profile_dim_provenance"
branch_labels = None
depends_on = None

ID = sa.String(length=36)
SHA = sa.String(length=64)
TS = sa.DateTime(timezone=True)

_STEPS_V1 = (
    "VALIDATE_SOURCE",
    "IMPORT_EPISODES",
    "NORMALIZE_EPISODE",
    "QUALITY_CHECK",
    "DETECT_OBJECTS",
    "BUILD_PREVIEW",
    "READY_FOR_REVIEW",
)
_STEPS_V21 = _STEPS_V1 + (
    "EXTRACT_KEYFRAMES",
    "VLM_PREANNOTATE_CLIP",
    "MERGE_CANDIDATE_ANNOTATIONS",
    "VALIDATE_AI_OUTPUT",
)


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def upgrade() -> None:
    # --- 1. tasks: per-clip fan-out ------------------------------------------------
    op.add_column(
        "tasks",
        sa.Column("unit_key", sa.String(length=64), nullable=False, server_default=""),
    )
    op.execute("DROP INDEX IF EXISTS ix_tasks_run_id_step")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_tasks_run_id_step_unit "
        "ON tasks (run_id, step, unit_key) WHERE run_id IS NOT NULL"
    )
    op.execute("ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_step_valid")
    op.execute("ALTER TABLE tasks DROP CONSTRAINT IF EXISTS step_valid")
    op.execute(
        "ALTER TABLE tasks ADD CONSTRAINT step_valid CHECK (step IN ("
        + _quoted(_STEPS_V21)
        + "))"
    )

    # --- 2. ai_clip_inputs ---------------------------------------------------------
    op.create_table(
        "ai_clip_inputs",
        sa.Column("id", ID, primary_key=True),
        sa.Column("episode_id", ID, sa.ForeignKey("episodes.id"), nullable=False),
        sa.Column("clip_id", SHA, nullable=False, unique=True),
        sa.Column("start_ns", sa.BigInteger(), nullable=False),
        sa.Column("end_ns", sa.BigInteger(), nullable=False),
        sa.Column("source_asset_sha256", SHA, nullable=False),
        sa.Column("preprocessing_sha256", SHA, nullable=False),
        sa.Column("frame_manifest_asset_id", ID, sa.ForeignKey("assets.id"), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("start_ns >= 0 AND end_ns > start_ns", name="clip_interval_valid"),
        sa.CheckConstraint("frame_count BETWEEN 1 AND 8", name="frame_count_valid"),
    )
    op.create_index("ix_ai_clip_inputs_episode", "ai_clip_inputs", ["episode_id", "start_ns"])

    # --- 3. model_inferences -------------------------------------------------------
    op.create_table(
        "model_inferences",
        sa.Column("id", ID, primary_key=True),
        sa.Column("run_id", ID, sa.ForeignKey("pipeline_runs.id"), nullable=False),
        sa.Column("task_id", ID, sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("clip_input_id", ID, sa.ForeignKey("ai_clip_inputs.id"), nullable=False),
        sa.Column("provider", sa.String(length=48), nullable=False),
        sa.Column("model_name", sa.String(length=96), nullable=False),
        sa.Column("mode", sa.String(length=12), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("prompt_sha256", SHA, nullable=False),
        sa.Column("request_sha256", SHA, nullable=False),
        sa.Column("output_asset_id", ID, sa.ForeignKey("assets.id"), nullable=True),
        sa.Column("output_sha256", SHA, nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("estimated_cost_usd", sa.Numeric(14, 6), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", TS, nullable=True),
        sa.CheckConstraint("mode IN ('LIVE','REPLAY')", name="mode_valid"),
        sa.CheckConstraint(
            "status IN ('QUEUED','RUNNING','SUCCEEDED','FAILED','BLOCKED')", name="status_valid"
        ),
        sa.UniqueConstraint(
            "run_id",
            "clip_input_id",
            "provider",
            "model_name",
            "prompt_sha256",
            name="inference_identity_unique",
        ),
    )
    op.create_index("ix_model_inferences_run", "model_inferences", ["run_id"])

    # --- 4. model_call_attempts ----------------------------------------------------
    op.create_table(
        "model_call_attempts",
        sa.Column("id", ID, primary_key=True),
        sa.Column("inference_id", ID, sa.ForeignKey("model_inferences.id"), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("provider_request_id", sa.String(length=128), nullable=True),
        sa.Column("response_sha256", SHA, nullable=True),
        sa.Column("usage_prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("usage_completion_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Numeric(14, 6), nullable=True),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("started_at", TS, nullable=False),
        sa.Column("finished_at", TS, nullable=True),
        sa.CheckConstraint("attempt_no BETWEEN 1 AND 3", name="attempt_no_valid"),
        sa.UniqueConstraint("inference_id", "attempt_no", name="inference_attempt_unique"),
    )

    # --- 5. candidate_action_annotations ------------------------------------------
    op.create_table(
        "candidate_action_annotations",
        sa.Column("id", ID, primary_key=True),
        sa.Column("annotation_set_id", ID, sa.ForeignKey("annotation_sets.id"), nullable=False),
        sa.Column("model_inference_id", ID, sa.ForeignKey("model_inferences.id"), nullable=False),
        sa.Column("clip_input_id", ID, sa.ForeignKey("ai_clip_inputs.id"), nullable=False),
        sa.Column("revision_id", ID, sa.ForeignKey("annotation_revisions.id"), nullable=True),
        sa.Column("action_index", sa.Integer(), nullable=False),
        sa.Column("verb", sa.String(length=32), nullable=False),
        sa.Column("noun", sa.String(length=64), nullable=False),
        sa.Column("start_ns", sa.BigInteger(), nullable=False),
        sa.Column("end_ns", sa.BigInteger(), nullable=False),
        sa.Column("evidence_frame_indexes", postgresql.JSONB(), nullable=False),
        sa.Column("uncertainty", sa.String(length=8), nullable=False),
        sa.Column("review_state", sa.String(length=16), nullable=False, server_default="PENDING"),
        sa.CheckConstraint("action_index BETWEEN 0 AND 11", name="action_index_valid"),
        sa.CheckConstraint("end_ns > start_ns", name="action_interval_valid"),
        sa.CheckConstraint("uncertainty IN ('low','medium','high')", name="uncertainty_valid"),
        sa.CheckConstraint(
            "review_state IN ('PENDING','IN_REVIEW','APPROVED','REJECTED')",
            name="review_state_valid",
        ),
        sa.UniqueConstraint("model_inference_id", "action_index", name="inference_action_unique"),
    )
    op.create_index(
        "ix_candidate_actions_annotation_set", "candidate_action_annotations", ["annotation_set_id"]
    )


def downgrade() -> None:
    bind = op.get_bind()
    # Additive tables with real rows are only dropped when they are provably empty;
    # otherwise the downgrade would destroy accepted AI evidence without a trace.
    for table in ("candidate_action_annotations", "model_call_attempts", "model_inferences", "ai_clip_inputs"):
        rows = bind.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one()
        if rows:
            raise RuntimeError(
                f"refusing to downgrade v21_002: {table} has {rows} rows; "
                "withdraw the AI evidence through a supported operation first"
            )

    op.drop_index("ix_candidate_actions_annotation_set", table_name="candidate_action_annotations")
    op.drop_table("candidate_action_annotations")
    op.drop_table("model_call_attempts")
    op.drop_index("ix_model_inferences_run", table_name="model_inferences")
    op.drop_table("model_inferences")
    op.drop_index("ix_ai_clip_inputs_episode", table_name="ai_clip_inputs")
    op.drop_table("ai_clip_inputs")

    op.execute("ALTER TABLE tasks DROP CONSTRAINT IF EXISTS step_valid")
    op.execute(
        "ALTER TABLE tasks ADD CONSTRAINT step_valid CHECK (step IN ("
        + _quoted(_STEPS_V1)
        + "))"
    )
    op.execute("DROP INDEX IF EXISTS ix_tasks_run_id_step_unit")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_tasks_run_id_step "
        "ON tasks (run_id, step) WHERE run_id IS NOT NULL"
    )
    op.drop_column("tasks", "unit_key")
