"""V2 D2-M6: scenario tags, coverage snapshots, collection-order lifecycle.

Revision ID: v2_005_feedback_coverage
Revises: v2_004_quality_v2

The pre-existing V1 order states are migrated to the V2 lifecycle:
OPEN -> PROPOSED, ACKNOWLEDGED -> IN_PROGRESS, CLOSED -> FULFILLED.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "v2_005_feedback_coverage"
down_revision = "v2_004_quality_v2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- source import provenance -------------------------------------------------
    op.add_column("source_imports", sa.Column("collection_order_id", sa.String(length=36), nullable=True))
    op.create_foreign_key(
        "fk_source_imports_collection_order_id",
        "source_imports",
        "collection_orders",
        ["collection_order_id"],
        ["id"],
    )

    # --- collection order lifecycle ----------------------------------------------
    op.add_column("collection_orders", sa.Column("created_by", sa.String(length=36), nullable=True))
    op.add_column("collection_orders", sa.Column("approved_by", sa.String(length=36), nullable=True))
    op.add_column("collection_orders", sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "collection_orders", sa.Column("approval_note", sa.Text(), server_default="", nullable=False)
    )
    op.add_column(
        "collection_orders",
        sa.Column("is_simulated_source", sa.Boolean(), server_default=sa.text("true"), nullable=False),
    )
    op.add_column(
        "collection_orders",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_foreign_key(
        "fk_collection_orders_created_by", "collection_orders", "api_keys", ["created_by"], ["id"]
    )
    op.create_foreign_key(
        "fk_collection_orders_approved_by", "collection_orders", "api_keys", ["approved_by"], ["id"]
    )
    op.execute("ALTER TABLE collection_orders DROP CONSTRAINT IF EXISTS ck_collection_orders_state_valid")
    op.execute(
        "UPDATE collection_orders SET state = CASE state "
        "WHEN 'OPEN' THEN 'PROPOSED' "
        "WHEN 'ACKNOWLEDGED' THEN 'IN_PROGRESS' "
        "WHEN 'CLOSED' THEN 'FULFILLED' "
        "ELSE state END"
    )
    op.execute(
        "ALTER TABLE collection_orders ADD CONSTRAINT ck_collection_orders_state_valid "
        "CHECK (state IN ('PROPOSED','APPROVED','IN_PROGRESS','FULFILLED','CANCELLED'))"
    )

    # --- fulfillment links, scenario tags and coverage snapshots ------------------
    op.create_table(
        "collection_order_episodes",
        sa.Column("order_id", sa.String(length=36), nullable=False),
        sa.Column("episode_id", sa.String(length=36), nullable=False),
        sa.Column("linked_by", sa.String(length=36), nullable=True),
        sa.Column("linked_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["order_id"], ["collection_orders.id"], name="fk_collection_order_episodes_order_id_collection_orders"
        ),
        sa.ForeignKeyConstraint(
            ["episode_id"], ["episodes.id"], name="fk_collection_order_episodes_episode_id_episodes"
        ),
        sa.ForeignKeyConstraint(
            ["linked_by"], ["api_keys.id"], name="fk_collection_order_episodes_linked_by_api_keys"
        ),
        sa.PrimaryKeyConstraint("order_id", "episode_id", name="pk_collection_order_episodes"),
        sa.UniqueConstraint("episode_id", name="episode_single_order_unique"),
    )
    op.create_index(
        "ix_collection_order_episodes_order_id", "collection_order_episodes", ["order_id"]
    )

    op.create_table(
        "scenario_tags",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("episode_id", sa.String(length=36), nullable=False),
        sa.Column("scenario_key", sa.String(length=64), nullable=False),
        sa.Column("basis", sa.String(length=20), nullable=False),
        sa.Column("source_evidence_asset_id", sa.String(length=36), nullable=True),
        sa.Column("verified_by", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["episode_id"], ["episodes.id"], name="fk_scenario_tags_episode_id_episodes"),
        sa.ForeignKeyConstraint(
            ["source_evidence_asset_id"], ["assets.id"], name="fk_scenario_tags_source_evidence_asset_id_assets"
        ),
        sa.ForeignKeyConstraint(
            ["verified_by"], ["api_keys.id"], name="fk_scenario_tags_verified_by_api_keys"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_scenario_tags"),
        sa.UniqueConstraint("episode_id", "scenario_key", "basis", name="episode_scenario_basis_unique"),
        sa.CheckConstraint(
            "basis IN ('SOURCE_DECLARED','HUMAN_VERIFIED','UNKNOWN')",
            name="ck_scenario_tags_basis_valid",
        ),
        sa.CheckConstraint(
            "scenario_key ~ '^[a-z0-9_]{3,64}$'", name="ck_scenario_tags_scenario_key_format"
        ),
    )

    op.create_table(
        "scenario_coverage_snapshots",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("scenario_key", sa.String(length=64), nullable=False),
        sa.Column("episode_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("tagging_basis", sa.String(length=20), server_default="UNKNOWN", nullable=False),
        sa.Column("source_digest", sa.String(length=64), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            name="fk_coverage_snapshots_version_id",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_scenario_coverage_snapshots"),
        sa.UniqueConstraint(
            "dataset_version_id", "scenario_key", "tagging_basis", name="version_scenario_basis_unique"
        ),
        sa.CheckConstraint(
            "tagging_basis IN ('HUMAN_VERIFIED','SOURCE_DECLARED','UNKNOWN')",
            name="ck_scenario_coverage_snapshots_basis_valid",
        ),
    )

    # --- gap analysis result asset (reuse the existing table, do not duplicate) ---
    op.add_column("gap_analyses", sa.Column("results_asset_id", sa.String(length=36), nullable=True))
    op.create_foreign_key(
        "fk_gap_analyses_results_asset_id_assets",
        "gap_analyses",
        "assets",
        ["results_asset_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_gap_analyses_results_asset_id_assets", "gap_analyses", type_="foreignkey")
    op.drop_column("gap_analyses", "results_asset_id")
    op.drop_table("scenario_coverage_snapshots")
    op.drop_table("scenario_tags")
    op.drop_index("ix_collection_order_episodes_order_id", table_name="collection_order_episodes")
    op.drop_table("collection_order_episodes")
    op.execute("ALTER TABLE collection_orders DROP CONSTRAINT IF EXISTS ck_collection_orders_state_valid")
    op.execute(
        "UPDATE collection_orders SET state = CASE state "
        "WHEN 'PROPOSED' THEN 'OPEN' "
        "WHEN 'IN_PROGRESS' THEN 'ACKNOWLEDGED' "
        "WHEN 'FULFILLED' THEN 'CLOSED' "
        "WHEN 'APPROVED' THEN 'OPEN' "
        "WHEN 'CANCELLED' THEN 'CLOSED' "
        "ELSE state END"
    )
    op.execute(
        "ALTER TABLE collection_orders ADD CONSTRAINT ck_collection_orders_state_valid "
        "CHECK (state IN ('OPEN','ACKNOWLEDGED','CLOSED'))"
    )
    op.drop_constraint("fk_collection_orders_approved_by", "collection_orders", type_="foreignkey")
    op.drop_constraint("fk_collection_orders_created_by", "collection_orders", type_="foreignkey")
    op.drop_column("collection_orders", "updated_at")
    op.drop_column("collection_orders", "is_simulated_source")
    op.drop_column("collection_orders", "approval_note")
    op.drop_column("collection_orders", "approved_at")
    op.drop_column("collection_orders", "approved_by")
    op.drop_column("collection_orders", "created_by")
    op.drop_constraint("fk_source_imports_collection_order_id", "source_imports", type_="foreignkey")
    op.drop_column("source_imports", "collection_order_id")
