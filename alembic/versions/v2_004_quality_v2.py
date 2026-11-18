"""V2 D2-M2: quality report/finding provenance fields.

Revision ID: v2_004_quality_v2
Revises: v2_003_conditional_operations
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "v2_004_quality_v2"
down_revision = "v2_003_conditional_operations"
branch_labels = None
depends_on = None

JSONB = postgresql.JSONB


def upgrade() -> None:
    op.add_column("quality_reports", sa.Column("profile_id", sa.String(length=80), nullable=True))
    op.add_column("quality_reports", sa.Column("profile_revision", sa.Integer(), nullable=True))
    op.add_column("quality_reports", sa.Column("config_sha256", sa.String(length=64), nullable=True))
    op.add_column(
        "quality_reports",
        sa.Column(
            "schema_version",
            sa.String(length=40),
            server_default="quality-report/1.0",
            nullable=False,
        ),
    )
    op.add_column("quality_findings", sa.Column("source_stream", sa.String(length=120), nullable=True))
    op.add_column(
        "quality_findings", sa.Column("source_clock_domain", sa.String(length=80), nullable=True)
    )
    op.add_column("quality_findings", sa.Column("method", sa.String(length=80), nullable=True))
    op.add_column(
        "quality_findings",
        sa.Column("limits_json", JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    )
    op.add_column("quality_findings", sa.Column("uncertainty_note", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("quality_findings", "uncertainty_note")
    op.drop_column("quality_findings", "limits_json")
    op.drop_column("quality_findings", "method")
    op.drop_column("quality_findings", "source_clock_domain")
    op.drop_column("quality_findings", "source_stream")
    op.drop_column("quality_reports", "schema_version")
    op.drop_column("quality_reports", "config_sha256")
    op.drop_column("quality_reports", "profile_revision")
    op.drop_column("quality_reports", "profile_id")
