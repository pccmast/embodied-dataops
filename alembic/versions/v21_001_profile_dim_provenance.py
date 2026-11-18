"""V2.1 M1: allow ``robot-profile/2.1`` documents in ``robot_profiles.config_json``.

Revision ID: v21_001_profile_dim_provenance
Revises: v2_006_widen_source_types

The V2.1 profile contract separates ``state.dimension`` from ``action.dimension`` and
makes ``limits`` optional (SPEC_V2.1 §4.2, §13).  ``config_json`` is JSONB, so the
document shape needs no DDL -- but the *generation* must be constrained, otherwise a
typo in a seed file silently registers an unreadable profile.

The constraint is added ``NOT VALID`` on purpose: legacy rows written before this
revision keep validating as before, while every new write is checked.  A later
migration can ``VALIDATE CONSTRAINT`` once the fleet is known-good.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "v21_001_profile_dim_provenance"
down_revision = "v2_006_widen_source_types"
branch_labels = None
depends_on = None

_ALLOWED = ("robot-profile/2.0", "robot-profile/2.1")


def upgrade() -> None:
    op.execute(
        "ALTER TABLE robot_profiles DROP CONSTRAINT IF EXISTS ck_robot_profiles_schema_generation"
    )
    op.execute(
        "ALTER TABLE robot_profiles ADD CONSTRAINT ck_robot_profiles_schema_generation "
        "CHECK (config_json ->> 'schema_version' IN "
        "('robot-profile/2.0','robot-profile/2.1')) NOT VALID"
    )
    # V2.1 profiles are looked up by (profile_id, revision) on every import; the unique
    # constraint already exists, this index makes the lookup explicit for planners.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_robot_profiles_lookup "
        "ON robot_profiles (profile_id, profile_revision)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    v21_rows = bind.execute(
        sa.text(
            "SELECT count(*) FROM robot_profiles WHERE config_json ->> 'schema_version' = 'robot-profile/2.1'"
        )
    ).scalar_one()
    if v21_rows:
        raise RuntimeError(
            "refusing to downgrade v21_001: robot-profile/2.1 rows exist; "
            "episodes bound to them would become unresolvable"
        )
    op.execute("DROP INDEX IF EXISTS ix_robot_profiles_lookup")
    op.execute("ALTER TABLE robot_profiles DROP CONSTRAINT IF EXISTS ck_robot_profiles_schema_generation")
