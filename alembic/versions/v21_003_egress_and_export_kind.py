"""V2.1: outbound-egress control and dataset export kind.

Revision ID: v21_003_egress_and_export_kind
Revises: v21_002_ai_inference_and_clips

Two additive columns that SPEC_V2.1 requires but V2 did not have:

``source_imports.egress_allowed`` (SPEC §8)
    Whether the bytes of this source may leave the deployment for a remote model call.
    **Default false**, and nothing in the AI pipeline can flip it: a live run refuses
    with ``SOURCE_EGRESS_NOT_ALLOWED`` instead of quietly sending data out.

``dataset_versions.export_kind`` (SPEC §14)
    ``ROBOT_LEROBOT_V3`` | ``EGO_ANNOTATION_BUNDLE`` | ``RESEARCH_SNAPSHOT``.  It makes
    "an ego video cannot be published as a robot control trajectory" a database fact
    rather than a convention.

Both are nullable-free with a server default, so existing rows keep working and the
migration is safe to run on a populated database.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "v21_003_egress_and_export_kind"
down_revision = "v21_002_ai_inference_and_clips"
branch_labels = None
depends_on = None

_KINDS = ("ROBOT_LEROBOT_V3", "EGO_ANNOTATION_BUNDLE", "RESEARCH_SNAPSHOT")


def upgrade() -> None:
    op.add_column(
        "source_imports",
        sa.Column("egress_allowed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "dataset_versions",
        sa.Column(
            "export_kind",
            sa.String(length=32),
            nullable=False,
            server_default="ROBOT_LEROBOT_V3",
        ),
    )
    op.execute("ALTER TABLE dataset_versions DROP CONSTRAINT IF EXISTS export_kind_valid")
    op.execute(
        "ALTER TABLE dataset_versions ADD CONSTRAINT export_kind_valid CHECK (export_kind IN ("
        + ", ".join(f"'{kind}'" for kind in _KINDS)
        + "))"
    )
    # A published ego bundle and a published robot version answer different questions;
    # the index keeps the "list this dataset's robot versions" query cheap.
    op.create_index(
        "ix_dataset_versions_export_kind", "dataset_versions", ["dataset_id", "export_kind"]
    )


def downgrade() -> None:
    bind = op.get_bind()
    non_default = bind.execute(
        sa.text("SELECT count(*) FROM dataset_versions WHERE export_kind <> 'ROBOT_LEROBOT_V3'")
    ).scalar_one()
    if non_default:
        raise RuntimeError(
            "refusing to downgrade v21_003: dataset versions with a non-default export_kind exist; "
            "their published identity would become ambiguous"
        )
    egress_rows = bind.execute(
        sa.text("SELECT count(*) FROM source_imports WHERE egress_allowed IS TRUE")
    ).scalar_one()
    if egress_rows:
        raise RuntimeError(
            "refusing to downgrade v21_003: sources with egress_allowed=true exist; "
            "dropping the column would erase a recorded authorisation"
        )
    op.drop_index("ix_dataset_versions_export_kind", table_name="dataset_versions")
    op.execute("ALTER TABLE dataset_versions DROP CONSTRAINT IF EXISTS export_kind_valid")
    op.drop_column("dataset_versions", "export_kind")
    op.drop_column("source_imports", "egress_allowed")
