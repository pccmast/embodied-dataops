"""V2 D2-M0: profile registry, native-stream evidence columns, canonical views.

Revision ID: v2_001_profile_native_streams
Revises: ccbc1672eacb
Create Date: 2026-10-29

Expand-only migration (ADR-0007):
* existing V1 rows keep ``canonical_schema_version='1.0'`` and ``profile_id=NULL``;
* no published asset/row is rewritten;
* downgrade refuses when any V2 object exists (no silent data loss).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "v2_001_profile_native_streams"
down_revision = "ccbc1672eacb"
branch_labels = None
depends_on = None

JSONB = postgresql.JSONB


def upgrade() -> None:
    op.create_table(
        "robot_profiles",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("profile_id", sa.String(length=80), nullable=False),
        sa.Column("profile_revision", sa.Integer(), nullable=False),
        sa.Column("authority", sa.String(length=32), nullable=False),
        sa.Column("source_evidence_asset_id", sa.String(length=36), nullable=True),
        sa.Column("config_json", JSONB(), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_evidence_asset_id"], ["assets.id"], name="fk_robot_profiles_source_evidence_asset_id_assets"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_robot_profiles"),
        sa.UniqueConstraint("profile_id", "profile_revision", name="profile_revision_unique"),
        sa.CheckConstraint(
            "authority IN ('SYNTHETIC_FIXTURE','SOURCE_DOCUMENTED','UNVERIFIED')",
            name="ck_robot_profiles_authority_valid",
        ),
        sa.CheckConstraint("profile_revision >= 1", name="ck_robot_profiles_revision_positive"),
    )

    op.add_column("episodes", sa.Column("profile_id", sa.String(length=80), nullable=True))
    op.add_column("episodes", sa.Column("profile_revision", sa.Integer(), nullable=True))
    op.add_column(
        "episodes",
        sa.Column("business_revision", sa.BigInteger(), server_default=sa.text("1"), nullable=False),
    )
    op.add_column(
        "episodes",
        sa.Column(
            "canonical_schema_version", sa.String(length=32), server_default=sa.text("'1.0'"), nullable=False
        ),
    )
    op.create_check_constraint(
        "ck_episodes_business_revision_positive", "episodes", "business_revision >= 1"
    )
    op.create_index("ix_episodes_profile", "episodes", ["profile_id", "profile_revision"], unique=False)

    op.add_column(
        "modalities",
        sa.Column("stream_index", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.add_column("modalities", sa.Column("native_rate_hz", sa.Float(), nullable=True))
    op.add_column("modalities", sa.Column("native_index_asset_id", sa.String(length=36), nullable=True))
    op.add_column("modalities", sa.Column("aligned_index_asset_id", sa.String(length=36), nullable=True))
    op.add_column("modalities", sa.Column("timestamp_provenance", sa.String(length=32), nullable=True))
    op.add_column(
        "modalities",
        sa.Column("clock_verified", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.create_foreign_key(
        "fk_modalities_native_index_asset_id_assets",
        "modalities",
        "assets",
        ["native_index_asset_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_modalities_aligned_index_asset_id_assets",
        "modalities",
        "assets",
        ["aligned_index_asset_id"],
        ["id"],
    )
    op.drop_constraint("episode_name_unique", "modalities", type_="unique")
    op.create_unique_constraint(
        "stream_unique", "modalities", ["episode_id", "name", "stream_index"]
    )
    op.create_check_constraint(
        "ck_modalities_stream_index_nonnegative", "modalities", "stream_index >= 0"
    )

    op.create_table(
        "canonical_views",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("episode_id", sa.String(length=36), nullable=False),
        sa.Column("canonical_version", sa.Integer(), nullable=False),
        sa.Column("view_kind", sa.String(length=32), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=False),
        sa.Column("metadata_asset_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["episode_id"], ["episodes.id"], name="fk_canonical_views_episode_id_episodes"),
        sa.ForeignKeyConstraint(
            ["metadata_asset_id"], ["assets.id"], name="fk_canonical_views_metadata_asset_id_assets"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_canonical_views"),
        sa.UniqueConstraint(
            "episode_id", "canonical_version", "view_kind", "config_sha256", name="view_config_unique"
        ),
        sa.CheckConstraint(
            "view_kind IN ('NATIVE','ALIGNED_TRAINING','PREVIEW')",
            name="ck_canonical_views_view_kind_valid",
        ),
    )


def downgrade() -> None:
    bind = op.get_bind()
    v2_objects = bind.execute(
        sa.text(
            "SELECT (SELECT count(*) FROM robot_profiles)"
            " + (SELECT count(*) FROM canonical_views)"
            " + (SELECT count(*) FROM episodes WHERE canonical_schema_version = '2.0'"
            "      OR profile_id IS NOT NULL)"
        )
    ).scalar_one()
    if v2_objects:
        raise RuntimeError(
            "refusing to downgrade v2_001: V2 profile/view/canonical objects exist; "
            "delete them via a supported migration instead of silently losing data"
        )
    op.drop_table("canonical_views")
    op.drop_constraint("ck_modalities_stream_index_nonnegative", "modalities", type_="check")
    op.drop_constraint("stream_unique", "modalities", type_="unique")
    op.create_unique_constraint("episode_name_unique", "modalities", ["episode_id", "name"])
    op.drop_constraint("fk_modalities_aligned_index_asset_id_assets", "modalities", type_="foreignkey")
    op.drop_constraint("fk_modalities_native_index_asset_id_assets", "modalities", type_="foreignkey")
    op.drop_column("modalities", "clock_verified")
    op.drop_column("modalities", "timestamp_provenance")
    op.drop_column("modalities", "aligned_index_asset_id")
    op.drop_column("modalities", "native_index_asset_id")
    op.drop_column("modalities", "native_rate_hz")
    op.drop_column("modalities", "stream_index")
    op.drop_index("ix_episodes_profile", table_name="episodes")
    op.drop_constraint("ck_episodes_business_revision_positive", "episodes", type_="check")
    op.drop_column("episodes", "canonical_schema_version")
    op.drop_column("episodes", "business_revision")
    op.drop_column("episodes", "profile_revision")
    op.drop_column("episodes", "profile_id")
    op.drop_table("robot_profiles")
