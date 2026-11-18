"""V2 D2-M4: business revisions, operation receipts and action effect fences.

Revision ID: v2_003_conditional_operations
Revises: v2_002_annotation_revisions
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "v2_003_conditional_operations"
down_revision = "v2_002_annotation_revisions"
branch_labels = None
depends_on = None


def _add_business_revision(table: str) -> None:
    """Idempotent-safe add for tables that did not get the column in v2_001."""
    bind = op.get_bind()
    columns = {
        row[0]
        for row in bind.execute(
            sa.text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :table AND table_schema = 'public'"
            ),
            {"table": table},
        )
    }
    if "business_revision" in columns:
        return
    op.add_column(
        table,
        sa.Column("business_revision", sa.BigInteger(), server_default=sa.text("1"), nullable=False),
    )
    op.create_check_constraint(f"ck_{table}_business_revision_positive", table, "business_revision >= 1")


def upgrade() -> None:
    _add_business_revision("tasks")
    _add_business_revision("evaluation_reports")

    op.create_table(
        "operation_receipts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=32), server_default="default", nullable=False),
        sa.Column("action_kind", sa.String(length=40), nullable=False),
        sa.Column("target_id", sa.String(length=36), nullable=False),
        sa.Column("source_business_revision", sa.BigInteger(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("effect_state", sa.String(length=20), server_default="COMMITTED", nullable=False),
        sa.Column("downstream_kind", sa.String(length=40), nullable=True),
        sa.Column("downstream_id", sa.String(length=36), nullable=True),
        sa.Column("error_code", sa.String(length=60), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_operation_receipts"),
        sa.UniqueConstraint("tenant_id", "idempotency_key", name="tenant_idempotency_unique"),
        sa.CheckConstraint(
            "action_kind IN ('REPROCESS_EPISODE','RETRY_FAILED_TASK','ANALYZE_EVALUATION_GAPS')",
            name="ck_operation_receipts_action_kind_valid",
        ),
        sa.CheckConstraint(
            "effect_state IN ('COMMITTED','IN_PROGRESS','FAILED')",
            name="ck_operation_receipts_effect_state_valid",
        ),
    )

    op.create_table(
        "action_effect_fences",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=32), server_default="default", nullable=False),
        sa.Column("action_kind", sa.String(length=40), nullable=False),
        sa.Column("target_id", sa.String(length=36), nullable=False),
        sa.Column("source_business_revision", sa.BigInteger(), nullable=False),
        sa.Column("operation_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["operation_id"], ["operation_receipts.id"], name="fk_action_effect_fences_operation_id_operation_receipts"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_action_effect_fences"),
        sa.UniqueConstraint(
            "tenant_id", "action_kind", "target_id", "source_business_revision", name="effect_fence_unique"
        ),
        sa.CheckConstraint(
            "action_kind IN ('REPROCESS_EPISODE','RETRY_FAILED_TASK','ANALYZE_EVALUATION_GAPS')",
            name="ck_action_effect_fences_action_kind_valid",
        ),
    )


def downgrade() -> None:
    op.drop_table("action_effect_fences")
    op.drop_table("operation_receipts")
    op.drop_constraint(
        "ck_evaluation_reports_business_revision_positive", "evaluation_reports", type_="check"
    )
    op.drop_column("evaluation_reports", "business_revision")
    op.drop_constraint("ck_tasks_business_revision_positive", "tasks", type_="check")
    op.drop_column("tasks", "business_revision")
