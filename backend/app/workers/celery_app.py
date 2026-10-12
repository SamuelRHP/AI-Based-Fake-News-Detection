"""Celery application instance and task broker configuration."""

from celery import Celery

from backend.app.core.config import backend_settings

celery_app = Celery(
    "fake_news_worker",
    broker=backend_settings.CELERY_BROKER_URL,
    backend=backend_settings.CELERY_RESULT_BACKEND,
    include=["backend.app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
)
