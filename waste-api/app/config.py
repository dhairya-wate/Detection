"""
app/config.py
=============
Application settings loaded from environment variables via Pydantic v2 BaseSettings.
All paths are kept as strings so inference.py remains import-time independent of FastAPI.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Override any field via environment variables."""

    # ── model & rules ──────────────────────────────────────────────────────
    MODEL_PATH: str = "models/best.onnx"
    RULES_PATH: str = "config/rules.json"

    # ── inference thresholds ───────────────────────────────────────────────
    CONF_THRES: float = 0.45
    IOU_THRES: float = 0.45

    # ── upload limits ──────────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = 10

    # ── ONNX Runtime ──────────────────────────────────────────────────────
    ONNX_THREADS: int = 2  # intra_op_num_threads

    # ── security ──────────────────────────────────────────────────────────
    API_KEY: str | None = None  # if set, X-API-Key header is required

    # ── CORS ──────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins, e.g. "http://localhost:3000,https://example.com"
    CORS_ORIGINS: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS_ORIGINS parsed into a list, filtering blank entries."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


# Module-level singleton – created once, shared across the app.
settings = Settings()
