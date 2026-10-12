"""Core configuration and utilities."""

from backend.app.core.config import backend_settings
from backend.app.core.logging import setup_logging

__all__ = ["backend_settings", "setup_logging"]
