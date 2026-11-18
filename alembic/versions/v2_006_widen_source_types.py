"""V2 D2-M1: allow MCAP_ROS2_PROFILE_V2 source imports.

Revision ID: v2_006_widen_source_types
Revises: v2_005_feedback_coverage
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "v2_006_widen_source_types"
down_revision = "v2_005_feedback_coverage"
branch_labels = None
depends_on = None

_V2_TYPES = ("LEROBOT_V3", "EGO_MP4", "MCAP_ROS2_PROFILE_V1", "MCAP_ROS2_PROFILE_V2")
_V1_TYPES = ("LEROBOT_V3", "EGO_MP4", "MCAP_ROS2_PROFILE_V1")


def upgrade() -> None:
    op.execute(
        "ALTER TABLE source_imports DROP CONSTRAINT IF EXISTS ck_source_imports_source_type_valid"
    )
    op.execute(
        "ALTER TABLE source_imports ADD CONSTRAINT ck_source_imports_source_type_valid "
        "CHECK (source_type IN " + str(_V2_TYPES).replace('(', '(').replace("'", "'") + ")"
    )


def downgrade() -> None:
    bind = op.get_bind()
    v2_rows = bind.execute(
        sa.text("SELECT count(*) FROM source_imports WHERE source_type = 'MCAP_ROS2_PROFILE_V2'")
    ).scalar_one()
    if v2_rows:
        raise RuntimeError(
            "refusing to downgrade v2_006: MCAP_ROS2_PROFILE_V2 imports exist; "
            "delete them via a supported migration first"
        )
    op.execute(
        "ALTER TABLE source_imports DROP CONSTRAINT IF EXISTS ck_source_imports_source_type_valid"
    )
    op.execute(
        "ALTER TABLE source_imports ADD CONSTRAINT ck_source_imports_source_type_valid "
        "CHECK (source_type IN " + str(_V1_TYPES) + ")"
    )
