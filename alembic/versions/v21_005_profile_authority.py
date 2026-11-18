"""V2.1: let ``robot_profiles.authority`` hold SOURCE_DOCUMENTED_PARTIAL.

Revision ID: v21_005_profile_authority
Revises: v21_004_model_attempt_cost_state

``robot-profile/2.1`` added ``SOURCE_DOCUMENTED_PARTIAL`` (SPEC_V2.1 §4.2) for the
real public snapshots whose shapes are documented but whose units/semantics are not
(AgiBot 16/16, LIBERO 8/7, utokyo 8/7, ALOHA 14/14).  The table-level check
constraint still only allowed the three 2.0 authorities, so registering any of those
revisions failed with a CheckViolation -- and ``qc-policy/2.0`` treats
"no ``robot_profiles`` row" as ``MISSING_ROBOT_PROFILE``, which made a correctly
bound V2.1 Episode look unbound and fail QC for the wrong reason.

The constraint is widened (never narrowed), so existing rows keep their authority.
"""

from __future__ import annotations

from alembic import op

revision = "v21_005_profile_authority"
down_revision = "v21_004_model_attempt_cost_state"
branch_labels = None
depends_on = None

#: Both names are dropped: the naming convention has prefixed the table name twice
#: in deployed databases, and a migration must not depend on which spelling won.
_CONSTRAINT_NAMES = (
    "ck_robot_profiles_ck_robot_profiles_authority_valid",
    "ck_robot_profiles_authority_valid",
)

_OLD = "('SYNTHETIC_FIXTURE','SOURCE_DOCUMENTED','UNVERIFIED')"
_NEW = "('SYNTHETIC_FIXTURE','SOURCE_DOCUMENTED','SOURCE_DOCUMENTED_PARTIAL','UNVERIFIED')"


def _set_authority_values(values: str) -> None:
    for name in _CONSTRAINT_NAMES:
        op.execute(f"ALTER TABLE robot_profiles DROP CONSTRAINT IF EXISTS {name}")
    op.execute(
        "ALTER TABLE robot_profiles ADD CONSTRAINT ck_robot_profiles_authority_valid "
        f"CHECK (authority IN {values})"
    )


def upgrade() -> None:
    _set_authority_values(_NEW)


def downgrade() -> None:
    # Not losslessly reversible: ``SOURCE_DOCUMENTED_PARTIAL`` has no 2.0 equivalent,
    # and silently rewriting it to ``SOURCE_DOCUMENTED`` would over-claim semantics.
    op.execute(
        "DO $$ BEGIN "
        "IF EXISTS (SELECT 1 FROM robot_profiles WHERE authority = 'SOURCE_DOCUMENTED_PARTIAL') "
        "THEN RAISE EXCEPTION "
        "'cannot narrow robot_profiles.authority: SOURCE_DOCUMENTED_PARTIAL rows exist'; "
        "END IF; END $$;"
    )
    _set_authority_values(_OLD)
