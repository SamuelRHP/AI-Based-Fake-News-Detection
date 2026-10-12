"""Health and liveness probes."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import backend_settings
from backend.app.db.session import get_db
from backend.app.schemas.health import HealthResponse
from fake_news_detector.models.registry import ModelRegistry

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check probe",
)
def get_health(db: Session = Depends(get_db)) -> HealthResponse:
    """Check database connection and model readiness."""
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    try:
        active_model = ModelRegistry.get_model().model_name
    except Exception:
        active_model = "unloaded"

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        environment="development" if backend_settings.DEBUG else "production",
        active_model=active_model,
        database_connected=db_ok,
        version="0.1.0",
    )
