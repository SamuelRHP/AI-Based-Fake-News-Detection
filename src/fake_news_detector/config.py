"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Fake News Detector application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts" / "models"

    # Decision Engine Thresholds
    FAKE_THRESHOLD_HIGH: float = Field(
        default=0.70,
        description="Probability threshold above which an article is classified as Fake.",
    )
    FAKE_THRESHOLD_LOW: float = Field(
        default=0.30,
        description="Probability threshold below which an article is classified as Real.",
    )

    # Model Configuration
    DEFAULT_MODEL_TYPE: Literal["baseline", "transformer"] = "baseline"
    TRANSFORMER_MODEL_NAME: str = "distilbert-base-uncased"
    MAX_SEQUENCE_LENGTH: int = 512
    RANDOM_STATE: int = 42

    # Scraper Settings
    REQUEST_TIMEOUT_SECONDS: int = 10
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FakeNewsDetectorBot/1.0"

    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "fake-news-detection"


settings = Settings()
