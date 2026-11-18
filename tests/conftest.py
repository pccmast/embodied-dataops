"""Shared pytest fixtures.

Test tiers
----------
* ``unit``        - no external services (local object store under tmp).
* ``integration`` - PostgreSQL 16 + Redis 7.4 (+ MinIO for a few cases).
* ``e2e``         - the full AT suite against a running stack (see tests/e2e).
* ``failure``     - fault injection against PostgreSQL/Redis.
* ``contract``    - official LeRobot API round-trip, OpenAPI snapshot.
* ``performance`` - benchmarks, excluded by default.

The PostgreSQL/Redis endpoints come from ``TEST_DATABASE_URL`` /
``TEST_REDIS_URL`` so CI and a local compose stack share the same fixtures.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# --- environment must be prepared before importing the application ----------
_OBJECT_ROOT = Path(tempfile.mkdtemp(prefix="dataops-test-objects-"))
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault(
    "TEST_DATABASE_URL", "postgresql+psycopg://dataops:dataops@localhost:5432/dataops_test"
)
os.environ.setdefault("TEST_REDIS_URL", "redis://localhost:6379/15")
os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
os.environ["STORAGE_BACKEND"] = "local"
os.environ["LOCAL_STORAGE_ROOT"] = str(_OBJECT_ROOT)
os.environ["DETECTOR"] = "fake"
os.environ["API_PUBLIC_BASE_URL"] = "http://testserver"
os.environ["CURSOR_SECRET"] = "test-cursor-secret"
os.environ["PIPELINE_VERSION"] = "1.0.0"

# --- Windows workaround: socket.socketpair() fallback ---------------------
# Python's Windows fallback (`_fallback_socketpair`) can fail with
# "Unexpected peer connection" when a loopback connection is intercepted or
# when localhost resolves differently; asyncio's self-pipe then breaks and
# Starlette TestClient hangs on lifespan startup.  Use an explicit IPv4
# loopback pair instead.  This is a test-infrastructure workaround only; it
# does not change production code paths.
if os.name == "nt":
    import socket as _socket

    try:
        _left, _right = _socket.socketpair()
    except OSError:
        def _windows_socketpair(family=None, type=_socket.SOCK_STREAM, proto=0):
            listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            client = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            client.connect(listener.getsockname())
            peer, _ = listener.accept()
            listener.close()
            return client, peer

        _socket.socketpair = _windows_socketpair  # type: ignore[assignment]
    else:
        _left.close()
        _right.close()

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from embodied_dataops.api.auth import PERMISSIONS, Role, generate_api_key  # noqa: E402
from embodied_dataops.config import get_settings, reset_settings_cache  # noqa: E402
from embodied_dataops.db.base import Base  # noqa: E402
from embodied_dataops.db.session import dispose_engine  # noqa: E402
from embodied_dataops.storage.base import LocalStorage, set_storage_for_tests  # noqa: E402


def _create_test_database() -> None:
    """Create the test database if the PostgreSQL server does not have it."""
    from urllib.parse import urlparse

    url = os.environ["TEST_DATABASE_URL"]
    database = urlparse(url).path.lstrip("/").split("?")[0]
    maintenance = url.rsplit("/", 1)[0] + "/postgres"
    admin_engine = create_engine(maintenance, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as connection:
        exists = connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": database}
        ).scalar()
        if not exists:
            connection.execute(text(f'CREATE DATABASE "{database}"'))
    admin_engine.dispose()


@pytest.fixture(scope="session")
def engine() -> Engine:
    _create_test_database()
    engine = create_engine(os.environ["TEST_DATABASE_URL"], pool_pre_ping=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="session")
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@pytest.fixture(autouse=True)
def clean_database(request):
    """Empty every business table before each test (task protocol needs commits).

    Tests marked ``no_db`` are pure in-process checks (schema, parsing, formatting) and
    opt out explicitly; they must not require a PostgreSQL instance.  The default is
    unchanged: every other test still gets a truncated database.
    """
    if request.node.get_closest_marker("no_db") is not None:
        yield
        return
    engine = request.getfixturevalue("engine")
    with engine.begin() as connection:
        table_names = ", ".join(f'"{table.name}"' for table in reversed(Base.metadata.sorted_tables))
        connection.execute(text(f"TRUNCATE {table_names} RESTART IDENTITY CASCADE"))
    yield


@pytest.fixture()
def db(session_factory: sessionmaker[Session]) -> Session:
    session = session_factory()
    try:
        yield session
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def settings():
    reset_settings_cache()
    dispose_engine()
    value = get_settings()
    yield value
    reset_settings_cache()
    dispose_engine()


@pytest.fixture()
def storage(tmp_path: Path) -> LocalStorage:
    settings = get_settings()
    backend = LocalStorage(
        settings.model_copy(update={"local_storage_root": tmp_path / "objects"})
    )
    set_storage_for_tests(backend)
    yield backend
    set_storage_for_tests(None)


@pytest.fixture()
def api_keys(db: Session) -> dict[str, str]:
    """Create one enabled API key per role; returns role -> plaintext key."""
    keys: dict[str, str] = {}
    for role in (Role.VIEWER, Role.OPERATOR, Role.REVIEWER, Role.ADMIN):
        raw, record = generate_api_key(role, name=f"test-{role.value}")
        db.add(record)
        keys[role.value] = raw
    db.commit()
    return keys


def auth_headers(raw_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {raw_key}"}


@pytest.fixture()
def viewer_headers(api_keys: dict[str, str]) -> dict[str, str]:
    return auth_headers(api_keys["viewer"])


@pytest.fixture()
def operator_headers(api_keys: dict[str, str]) -> dict[str, str]:
    return auth_headers(api_keys["operator"])


@pytest.fixture()
def reviewer_headers(api_keys: dict[str, str]) -> dict[str, str]:
    return auth_headers(api_keys["reviewer"])


@pytest.fixture()
def admin_headers(api_keys: dict[str, str]) -> dict[str, str]:
    return auth_headers(api_keys["admin"])


@pytest.fixture()
def client(engine: Engine, storage: LocalStorage) -> TestClient:
    from embodied_dataops.api.app import create_app

    dispose_engine()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def redis_client():
    try:
        import redis as redis_lib

        client = redis_lib.Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
        client.ping()
        client.flushdb()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Redis is not available: {exc}")
    yield client
    client.flushdb()
    client.close()


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure in-process unit tests")
    config.addinivalue_line("markers", "integration: needs PostgreSQL/Redis")
    config.addinivalue_line("markers", "e2e: end-to-end acceptance test")
    config.addinivalue_line("markers", "failure: failure injection test")
    config.addinivalue_line("markers", "contract: external contract test")
    config.addinivalue_line("markers", "performance: benchmark")
    config.addinivalue_line("markers", "ros2: needs the ROS 2 Jazzy worker")
    config.addinivalue_line("markers", "vision: needs real YOLO11n weights")
    config.addinivalue_line(
        "markers",
        "concurrency: two workers competing on real row locks / unique keys; the release gate",
    )


__all__ = ["PERMISSIONS", "auth_headers"]
