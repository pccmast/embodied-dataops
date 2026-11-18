"""V2 D2-M3: annotation revisions, edits and dataset snapshot link.

Revision ID: v2_002_annotation_revisions
Revises: v2_001_profile_native_streams
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "v2_002_annotation_revisions"
down_revision = "v2_001_profile_native_streams"
branch_labels = None
depends_on = None

JSONB = postgresql.JSONB


def upgrade() -> None:
    op.add_column(
        "annotation_sets",
        sa.Column("current_revision_no", sa.Integer(), server_default=sa.text("1"), nullable=False),
    )
    op.add_column("annotation_sets", sa.Column("approved_revision_id", sa.String(length=36), nullable=True))
    op.create_check_constraint(
        "ck_annotation_sets_revision_positive", "annotation_sets", "current_revision_no >= 1"
    )

    op.create_table(
        "annotation_revisions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("annotation_set_id", sa.String(length=36), nullable=False),
        sa.Column("revision_no", sa.Integer(), nullable=False),
        sa.Column("base_revision_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_asset_id", sa.String(length=36), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=20), server_default="DRAFT", nullable=False),
        sa.Column("created_by", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("approved_by", sa.String(length=36), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["annotation_set_id"], ["annotation_sets.id"], name="fk_annotation_revisions_annotation_set_id_annotation_sets"
        ),
        sa.ForeignKeyConstraint(
            ["base_revision_id"], ["annotation_revisions.id"], name="fk_annotation_revisions_base_revision_id_annotation_revisions"
        ),
        sa.ForeignKeyConstraint(
            ["artifact_asset_id"], ["assets.id"], name="fk_annotation_revisions_artifact_asset_id_assets"
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["api_keys.id"], name="fk_annotation_revisions_created_by_api_keys"
        ),
        sa.ForeignKeyConstraint(
            ["approved_by"], ["api_keys.id"], name="fk_annotation_revisions_approved_by_api_keys"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_annotation_revisions"),
        sa.UniqueConstraint("annotation_set_id", "revision_no", name="set_revision_unique"),
        sa.CheckConstraint("revision_no >= 1", name="ck_annotation_revisions_revision_positive"),
        sa.CheckConstraint(
            "state IN ('DRAFT','IN_REVIEW','APPROVED','REJECTED')",
            name="ck_annotation_revisions_state_valid",
        ),
    )
    op.create_index(
        "ix_annotation_revisions_set", "annotation_revisions", ["annotation_set_id", "revision_no"]
    )

    op.create_table(
        "annotation_revision_edits",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("revision_id", sa.String(length=36), nullable=False),
        sa.Column("actor_key_id", sa.String(length=36), nullable=False),
        sa.Column("change_summary_json", JSONB(), nullable=False),
        sa.Column("before_sha256", sa.String(length=64), nullable=False),
        sa.Column("after_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["revision_id"], ["annotation_revisions.id"], name="fk_annotation_revision_edits_revision_id_annotation_revisions"
        ),
        sa.ForeignKeyConstraint(
            ["actor_key_id"], ["api_keys.id"], name="fk_annotation_revision_edits_actor_key_id_api_keys"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_annotation_revision_edits"),
    )
    op.create_index(
        "ix_annotation_revision_edits_revision_id", "annotation_revision_edits", ["revision_id"]
    )

    op.create_foreign_key(
        "fk_annotation_sets_approved_revision_id",
        "annotation_sets",
        "annotation_revisions",
        ["approved_revision_id"],
        ["id"],
        use_alter=True,
    )
    op.add_column(
        "dataset_version_episodes", sa.Column("annotation_revision_id", sa.String(length=36), nullable=True)
    )
    op.create_foreign_key(
        "fk_dataset_version_episodes_annotation_revision_id",
        "dataset_version_episodes",
        "annotation_revisions",
        ["annotation_revision_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_dataset_version_episodes_annotation_revision_id",
        "dataset_version_episodes",
        type_="foreignkey",
    )
    op.drop_column("dataset_version_episodes", "annotation_revision_id")
    op.drop_constraint("fk_annotation_sets_approved_revision_id", "annotation_sets", type_="foreignkey")
    op.drop_index("ix_annotation_revision_edits_revision_id", table_name="annotation_revision_edits")
    op.drop_table("annotation_revision_edits")
    op.drop_index("ix_annotation_revisions_set", table_name="annotation_revisions")
    op.drop_table("annotation_revisions")
    op.drop_constraint(
        "ck_annotation_sets_revision_positive", "annotation_sets", type_="check"
    )
    op.drop_column("annotation_sets", "approved_revision_id")
    op.drop_column("annotation_sets", "current_revision_no")
