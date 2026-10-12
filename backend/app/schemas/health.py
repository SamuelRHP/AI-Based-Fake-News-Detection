"""System health and status schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """API health status report."""

    status: str
    environment: str
    active_model: str
    database_connected: bool
    version: str
