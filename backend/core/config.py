"""
visp.core.config
~~~~~~~~~~~~~~~~
All application settings, sourced from environment variables or .env file.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelBackend(str, Enum):
    MVIT = "mvit"
    VIVIT = "vivit"
    R2PLUS1D = "r2plus1d"
    ONNX = "onnx"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Application ───────────────────────────────────────────────
    app_name: str = "VISP — Video Intelligence & Safety Platform"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Auth ──────────────────────────────────────────────────────
    api_key: str = Field(..., description="Secret key for API authentication")

    # ── Model ─────────────────────────────────────────────────────
    model_backend: ModelBackend = ModelBackend.MVIT
    onnx_model_path: str = "models/visp_model.onnx"
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    clip_length: int = Field(default=16, description="Number of frames per inference window")
    frame_skip: int = Field(default=2, description="Process every Nth frame")
    device: str = "cpu"  # "cuda" | "cpu" | "mps"

    # ── Streaming ─────────────────────────────────────────────────
    max_concurrent_streams: int = Field(default=8, ge=1)

    # ── Redis ─────────────────────────────────────────────────────
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")  # type: ignore[assignment]
    event_ttl_seconds: int = 86_400  # 24h

    # ── Database ──────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://visp:visp@localhost:5432/visp"

    # ── Alerts ────────────────────────────────────────────────────
    alert_webhook_url: AnyHttpUrl | None = None
    alert_email_to: str | None = None
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    slack_bot_token: str | None = None

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()  # type: ignore[call-arg]
