"""Logging configuration for API and background workers."""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging across services."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
