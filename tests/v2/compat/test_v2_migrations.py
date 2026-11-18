"""V2 migration tests (DO2-01/04).

Each test creates an independent PostgreSQL database via the maintenance database
from ``TEST_DATABASE_URL`` and runs Alembic programmatically.  The shared fixture
database from ``tests/conftest.py`` is not touched.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
INITIAL_REVISION = "ccbc1672eacb"


def _head_revision() -> str:
    """The single head the migration graph declares, read from the scripts themselves.

    Hardcoding the head here made this test fail the moment a newer revision landed
    (``v21_001`` and later), even though "upgrade to head twice is a no-op" was still
    true.  The property under test is reentrancy, not a particular revision string.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    heads = ScriptDirectory.from_config(config).get_heads()
    assert len(heads) == 1, f"the migration graph must have exactly one head, got {heads}"
    return heads[0]


@pytest.fixture(autouse=True)
def _restore_database_url():
    """Restore DATABASE_URL after Alembic tests (they point it at temporary DBs).

    Without this, later tests in the same pytest process would connect to a
    dropped temporary database, because Alembic reads ``DATABASE_URL`` from the
    environment.
    """
    import os

    original = os.environ.get("DATABASE_URL")
    yield
    if original is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = original
    from embodied_dataops.db.session import dispose_engine

    dispose_engine()


def _maintenance_url() -> str:
    import os

    url = os.environ["TEST_DATABASE_URL"]
    return url.rsplit("/", 1)[0] + "/postgres"


def _create_database(name: str) -> str:
    admin = create_engine(_maintenance_url(), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    base = _maintenance_url().rsplit("/", 1)[0]
    return f"{base}/{name}"


def _drop_database(name: str) -> None:
    admin = create_engine(_maintenance_url(), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
    admin.dispose()


def _alembic_config(url: str):
    import os

    from alembic.config import Config

    os.environ["DATABASE_URL"] = url
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def _upgrade(url: str, revision: str) -> None:
    from alembic import command

    command.upgrade(_alembic_config(url), revision)


def _downgrade(url: str, revision: str) -> None:
    from alembic import command

    command.downgrade(_alembic_config(url), revision)


def _columns(url: str, table: str) -> set[str]:
    engine = create_engine(url)
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name = :table"),
            {"table": table},
        ).scalars()
        values = set(rows)
    engine.dispose()
    return values


def _tables(url: str) -> set[str]:
    engine = create_engine(url)
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'")).scalars()
        values = set(rows)
    engine.dispose()
    return values


def test_empty_database_upgrade_is_reentrant_and_creates_v2_schema():
    name = f"dataops_v2_mig_{uuid.uuid4().hex[:8]}"
    url = _create_database(name)
    try:
        _upgrade(url, "head")
        assert {"robot_profiles", "canonical_views"} <= _tables(url)
        episode_columns = _columns(url, "episodes")
        assert {
            "profile_id",
            "profile_revision",
            "business_revision",
            "canonical_schema_version",
        } <= episode_columns
        modality_columns = _columns(url, "modalities")
        assert {
            "stream_index",
            "native_rate_hz",
            "native_index_asset_id",
            "aligned_index_asset_id",
            "timestamp_provenance",
            "clock_verified",
        } <= modality_columns
        _upgrade(url, "head")  # second run is a no-op, not an error
        engine = create_engine(url)
        with engine.connect() as connection:
            revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        engine.dispose()
        assert revision == _head_revision()
    finally:
        _drop_database(name)


def test_v1_database_upgrade_preserves_legacy_rows_and_assets():
    name = f"dataops_v2_legacy_{uuid.uuid4().hex[:8]}"
    url = _create_database(name)
    try:
        _upgrade(url, INITIAL_REVISION)
        engine = create_engine(url)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO assets (id,sha256,byte_size,media_type,object_key,origin,"
                    "logical_name,meta_json,created_at) VALUES "
                    "('asset-legacy',repeat('a',64),123,'video/mp4','raw/sha256/legacy','raw',"
                    "'source','{}',now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO source_imports (id,source_type,upload_state,asset_id,data_origin,"
                    "license,filename,size_bytes,sha256,mapping_json,created_at) VALUES "
                    "('imp-legacy','EGO_MP4','READY','asset-legacy','synthetic','x','a.mp4',123,"
                    "repeat('a',64),'{}',now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO episodes (id,source_import_id,source_episode_index,episode_type,"
                    "task_label,duration_ns,status,canonical_version,canonical_sha256,quality_status,"
                    "sample_count,data_origin,created_at,updated_at) VALUES "
                    "('ep-legacy','imp-legacy',0,'ego_video_episode','legacy',10000000000,"
                    "'READY_FOR_REVIEW',1,repeat('b',64),'PASS',100,'synthetic',now(),now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO pipeline_runs (id,episode_id,pipeline_version,state,"
                    "cancel_requested,created_at) VALUES "
                    "('run-legacy','ep-legacy','1.0.0','SUCCEEDED',false,now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO quality_reports (id,episode_id,run_id,policy_version,status,"
                    "summary_json,created_at) VALUES "
                    "('qr-legacy','ep-legacy','run-legacy','qc-policy/1.0','PASS','{}',now())"
                )
            )
            connection.execute(
                text("INSERT INTO datasets (id,slug,description,created_at) VALUES ('ds-legacy','legacy-ds','x',now())")
            )
            connection.execute(
                text(
                    "INSERT INTO dataset_versions (id,dataset_id,version,state,manifest_sha256,"
                    "description,created_at) VALUES "
                    "('dv-legacy','ds-legacy',1,'PUBLISHED',repeat('c',64),'x',now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO dataset_version_episodes (version_id,episode_id,canonical_digest,"
                    "quality_report_id,position) VALUES "
                    "('dv-legacy','ep-legacy',repeat('d',64),'qr-legacy',0)"
                )
            )
        engine.dispose()

        _upgrade(url, "head")

        engine = create_engine(url)
        with engine.connect() as connection:
            episode = connection.execute(
                text(
                    "SELECT canonical_version, canonical_schema_version, business_revision, "
                    "profile_id FROM episodes WHERE id='ep-legacy'"
                )
            ).one()
            assert episode.canonical_version == 1
            assert episode.canonical_schema_version == "1.0"  # never back-filled
            assert episode.business_revision == 1
            assert episode.profile_id is None  # no guessed profile

            asset = connection.execute(
                text("SELECT sha256, object_key FROM assets WHERE id='asset-legacy'")
            ).one()
            assert asset.sha256 == "a" * 64
            assert asset.object_key == "raw/sha256/legacy"

            version = connection.execute(
                text("SELECT state, manifest_sha256 FROM dataset_versions WHERE id='dv-legacy'")
            ).one()
            assert version.state == "PUBLISHED"
            assert version.manifest_sha256 == "c" * 64
        engine.dispose()
    finally:
        _drop_database(name)


def test_downgrade_refuses_when_v2_objects_exist():
    name = f"dataops_v2_guard_{uuid.uuid4().hex[:8]}"
    url = _create_database(name)
    try:
        _upgrade(url, "head")
        engine = create_engine(url)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO robot_profiles (id,profile_id,profile_revision,authority,"
                    "config_json,config_sha256) VALUES "
                    "('prof-1','arm6_joint_position_v1',1,'SYNTHETIC_FIXTURE','{}',repeat('e',64))"
                )
            )
        engine.dispose()
        with pytest.raises(RuntimeError, match="refusing to downgrade"):
            _downgrade(url, INITIAL_REVISION)
        # V1 data remains queryable after the refused downgrade
        engine = create_engine(url)
        with engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM robot_profiles")).scalar_one() == 1
        engine.dispose()
    finally:
        _drop_database(name)
