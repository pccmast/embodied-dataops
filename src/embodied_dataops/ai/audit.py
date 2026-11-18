"""Durable model-call audit (SPEC_V2.1 §5.4, §13).

A failed clip raises :class:`~embodied_dataops.errors.TerminalTaskError`, and the
workflow engine reacts by rolling the task transaction back.  Anything the handler
wrote in that transaction -- ``model_inferences`` status, ``model_call_attempts``
rows, the estimated cost -- disappears with it, even though the remote provider was
already called and may already have billed the request.

This module gives the AI branch a **second, immediately committed** writer for
exactly the rows that describe the outbound call:

* ``model_inferences`` identity/reservation (``RUNNING`` + estimated cost),
* every ``model_call_attempts`` row,
* the terminal ``FAILED``/``BLOCKED`` state and its error code.

What it deliberately does **not** write is a success: the committed product
(``output_asset_id``) is set by the task transaction, so "one valid committed
product per inference" stays in one atomic transaction with the candidate rows.
If that transaction is lost, the inference stays ``RUNNING`` -- a conservative
"reserved, not settled" state that a retry reuses under the same unique key.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from sqlalchemy import Connection
from sqlalchemy.orm import Session, sessionmaker


def open_audit_session(session: Session) -> Session:
    """Open a session on the same database that commits independently of ``session``.

    The caller's session is usually bound to an ``Engine``; when it is bound to an
    explicit :class:`~sqlalchemy.Connection` (tests), the connection's engine is used
    so the audit write does not piggy-back on the caller's transaction.
    """
    bind = session.get_bind()
    engine = bind.engine if isinstance(bind, Connection) else bind
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return factory()


@contextlib.contextmanager
def audit_scope(session: Session) -> Iterator[Session]:
    """Yield an independent session and always close it, even on failure."""
    audit = open_audit_session(session)
    try:
        yield audit
    finally:
        with contextlib.suppress(Exception):
            audit.rollback()
        audit.close()


__all__ = ["audit_scope", "open_audit_session"]
