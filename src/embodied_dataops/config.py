"""Runtime configuration.

All values come from environment variables / `.env`; nothing is hard-coded in
business logic.  The settings object is intentionally explicit (no magic
prefixes) so docker-compose, CI and tests can share the same names.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    # --- runtime -----------------------------------------------------------
    environment: str = "local"
    service_name: str = "api"
    log_level: str = "INFO"

    # --- PostgreSQL (task truth source) ------------------------------------
    database_url: str = "postgresql+psycopg://dataops:dataops@localhost:5432/dataops"
    db_pool_size: int = 10
    db_max_overflow: int = 10
    db_echo: bool = False

    # --- Redis (wakeup hints only) ----------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- object storage ----------------------------------------------------
    storage_backend: Literal["s3", "local"] = "s3"
    s3_bucket: str = "dataops"
    s3_endpoint_internal: str = "http://minio:9000"
    s3_endpoint_public: str = "http://localhost:9000"
    s3_access_key: str = "dataops"
    s3_secret_key: str = "dataops-secret"
    s3_region: str = "us-east-1"
    local_storage_root: Path = Path("./var/objects")

    api_public_base_url: str = "http://localhost:8000"
    ui_api_base_url: str = "http://localhost:8000"
    cursor_secret: str = "dataops-cursor-dev-secret"

    # --- upload / import limits -------------------------------------------
    max_upload_bytes: int = 2 * 1024**3
    max_unpacked_bytes: int = 4 * 1024**3
    max_tar_entries: int = 10_000
    upload_url_ttl_seconds: int = 900
    download_url_ttl_seconds: int = 600
    max_import_seconds: int = 600

    # --- task protocol -----------------------------------------------------
    lease_seconds: int = 60
    heartbeat_seconds: int = 15
    reaper_interval_seconds: int = 10
    dispatcher_interval_seconds: int = 1
    max_attempts: int = 3
    retry_backoff_seconds: tuple[float, ...] = (5.0, 20.0)
    worker_pool: str = "cpu"
    worker_concurrency: int = 2
    worker_id: str | None = None
    worker_http_port: int = 0

    # --- pipeline / quality ------------------------------------------------
    pipeline_version: str = "1.0.0"
    qc_policy_version: str = "qc-policy/1.0"
    canonical_staging_ttl_hours: int = 24

    # --- detection ---------------------------------------------------------
    detector: Literal["fake", "yolo11n"] = "fake"
    yolo_weights_path: Path = Path("./var/models/yolo11n.pt")
    yolo_weights_sha256: str = ""
    yolo_conf: float = 0.35
    yolo_iou: float = 0.5
    yolo_device: str = "cpu"

    # --- optional P1 providers --------------------------------------------
    sam2_enabled: bool = False
    vlm_enabled: bool = False
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""

    # --- V2.1 AI pre-annotation (SPEC_V2.1 §2, §5.4) -----------------------
    #: ``disabled`` keeps the CPU pipeline runnable without any model; ``replay`` runs the
    #: same code path against recorded answers and can never satisfy a live acceptance
    #: item; ``dashscope`` performs real remote calls and requires an API key.
    vlm_provider: Literal["disabled", "replay", "dashscope"] = "disabled"
    vlm_prompt_version: str = "ego-action-prompt/2.1"
    vlm_prompt_path: Path = Path("./contracts/EGO_ACTION_PROMPT_V21.md")
    vlm_max_images_per_clip: int = 8
    vlm_max_side_px: int = 768
    vlm_jpeg_quality: int = 80
    vlm_timeout_s: int = 60
    vlm_max_attempts: int = 3
    vlm_concurrency: int = 2
    vlm_max_output_tokens: int = 512
    vlm_budget_usd_per_run: float = 2.0
    #: A ``model_inferences`` row left RUNNING this long is a reservation whose worker
    #: died between reserving and settling.  It is released by the reaper, otherwise a
    #: killed worker would hold budget on that run forever.  Must stay above the
    #: VLM_PREANNOTATE_CLIP wall clock (900 s) so a live call is never reaped.
    vlm_stale_inference_seconds: int = 1800
    #: Price snapshot used for cost estimates.  ``None`` means "unknown" and the report
    #: says so instead of inventing a zero (SPEC_V2.1 §5.4).
    vlm_price_per_million_prompt_usd: float | None = None
    vlm_price_per_million_completion_usd: float | None = None
    vlm_price_snapshot_note: str = ""
    #: Recorded responses for ``VLM_PROVIDER=replay`` (one JSON per clip id).
    vlm_replay_dir: Path = Path("./var/vlm-replay")

    # --- observability -----------------------------------------------------
    prometheus_enabled: bool = True

    # --- derived paths -----------------------------------------------------
    scratch_dir: Path = Path("./var/scratch")
    model_cache_dir: Path = Path("./var/models")

    @field_validator(
        "local_storage_root",
        "yolo_weights_path",
        "scratch_dir",
        "model_cache_dir",
        "vlm_prompt_path",
        "vlm_replay_dir",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    @field_validator(
        "vlm_price_per_million_prompt_usd",
        "vlm_price_per_million_completion_usd",
        mode="before",
    )
    @classmethod
    def _blank_price_means_unknown(cls, value: object) -> object:
        """``.env`` 里留空的价钱表示"未知"，不是"零"。

        ``.env.example`` 明确要求未知时留空，但空字符串会被 pydantic 当成无法解析的
        float 并让整个进程启动失败。这里把空串收敛成 ``None``，报告才能如实写
        "cost unknown" 而不是编一个 0。
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def sync_database_url(self) -> str:
        return self.database_url

    def ensure_directories(self) -> None:
        self.local_storage_root.mkdir(parents=True, exist_ok=True)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    def public_url(self, internal_url: str) -> str:
        """Rewrite an internal MinIO endpoint to the browser-reachable one."""
        internal = self.s3_endpoint_internal.rstrip("/")
        public = self.s3_endpoint_public.rstrip("/")
        if internal and internal_url.startswith(internal):
            return public + internal_url[len(internal) :]
        return internal_url


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
