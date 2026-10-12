"""API v1 router registry."""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import batch, explain, feedback, health, predict

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(predict.router, tags=["Predictions"])
api_router.include_router(explain.router, tags=["Explainability"])
api_router.include_router(batch.router, tags=["Batch Processing"])
api_router.include_router(feedback.router, tags=["Human Feedback"])
