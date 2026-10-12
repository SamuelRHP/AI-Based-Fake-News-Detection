"""Backend application configuration via Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Central configuration for FastAPI and Celery worker."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "AI-Based Fake News Detection API"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./fake_news.db",
        description="SQLAlchemy database connection string.",
    )

    # Redis & Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # Inference defaults
    DEFAULT_MODEL_TYPE: str = "baseline"
    EXPLAINABILITY_NUM_FEATURES: int = 10


backend_settings = BackendSettings()
