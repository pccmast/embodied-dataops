"""Engine/session lifecycle.  PostgreSQL is the only task truth source."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from embodied_dataops.config import Settings, get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_engine_url: str | None = None


def build_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    return create_engine(
        settings.sync_database_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        future=True,
        echo=settings.db_echo,
    )


def get_engine() -> Engine:
    global _engine, _engine_url
    if _engine is None:
        settings = get_settings()
        _engine = build_engine(settings)
        _engine_url = settings.sync_database_url
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on exception, always close."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """Test helper: drop the process-wide engine/session factory."""
    global _engine, _session_factory, _engine_url
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    _engine_url = None


def reset_for_tests(database_url: str | None = None) -> None:
    from embodied_dataops.config import reset_settings_cache

    dispose_engine()
    if database_url is not None:
        import os

        os.environ["DATABASE_URL"] = database_url
    reset_settings_cache()
