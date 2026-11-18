"""V2.1: per-attempt cost state for remote model calls.

Revision ID: v21_004_model_attempt_cost_state
Revises: v21_003_egress_and_export_kind

``model_call_attempts`` recorded ``estimated_cost_usd IS NULL`` for every failed
attempt, which conflates two very different facts:

* the provider rejected the request before running anything (401/403/422) -- nothing
  was billed;
* the transport timed out or the response body was lost -- the provider may have
  run the inference and billed it, we simply cannot see the amount.

SPEC_V2.1 §5.4 requires that distinction to be explicit ("lost response 可导致重复
付费 ... 费用只可标 ``UNKNOWN_UPSTREAM_CHARGE``").  The new column is additive, has a
server default and never rewrites an existing row's meaning.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "v21_004_model_attempt_cost_state"
down_revision = "v21_003_egress_and_export_kind"
branch_labels = None
depends_on = None

_STATES = ("KNOWN", "UNKNOWN_UPSTREAM_CHARGE", "NOT_BILLED")


def upgrade() -> None:
    op.add_column(
        "model_call_attempts",
        sa.Column(
            "cost_state",
            sa.String(length=24),
            nullable=False,
            server_default="NOT_BILLED",
        ),
    )
    # Backfill only what the old rows can actually prove:
    #   * a 200 response with a recorded cost is KNOWN;
    #   * a 200 response without a price snapshot is an unknown charge;
    #   * anything without a 2xx status was not billed (the previous convention).
    op.execute(
        "UPDATE model_call_attempts SET cost_state = 'KNOWN' "
        "WHERE response_status = 200 AND estimated_cost_usd IS NOT NULL"
    )
    op.execute(
        "UPDATE model_call_attempts SET cost_state = 'UNKNOWN_UPSTREAM_CHARGE' "
        "WHERE response_status = 200 AND estimated_cost_usd IS NULL"
    )
    op.execute("ALTER TABLE model_call_attempts DROP CONSTRAINT IF EXISTS cost_state_valid")
    op.execute(
        "ALTER TABLE model_call_attempts ADD CONSTRAINT cost_state_valid CHECK (cost_state IN ("
        + ", ".join(f"'{state}'" for state in _STATES)
        + "))"
    )


def downgrade() -> None:
    bind = op.get_bind()
    unknown = bind.execute(
        sa.text("SELECT count(*) FROM model_call_attempts WHERE cost_state = 'UNKNOWN_UPSTREAM_CHARGE'")
    ).scalar_one()
    if unknown:
        raise RuntimeError(
            "refusing to downgrade v21_004: attempts with an unknown upstream charge exist; "
            "dropping the column would turn a possible bill into a silent zero"
        )
    op.execute("ALTER TABLE model_call_attempts DROP CONSTRAINT IF EXISTS cost_state_valid")
    op.drop_column("model_call_attempts", "cost_state")
