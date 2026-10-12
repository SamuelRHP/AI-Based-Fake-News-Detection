"""Celery background tasks for async batch inference and training."""

import logging
from typing import Any

from backend.app.schemas.article import ArticleInput
from backend.app.services.prediction_service import PredictionService
from backend.app.workers.celery_app import celery_app
from fake_news_detector.pipelines.train import run_training_pipeline

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.predict_batch_task", bind=True)
def predict_batch_task(
    self: Any,
    items: list[dict[str, Any]],
    model_type: str | None = None,
) -> dict[str, Any]:
    """Execute asynchronous batch predictions across multiple articles."""
    logger.info(f"Starting batch prediction task {self.request.id} for {len(items)} items.")
    service = PredictionService()
    results: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        item_id = item.get("id", f"item-{index + 1}")
        try:
            inp = ArticleInput(
                text=item.get("text"),
                url=item.get("url"),
                title=item.get("title"),
                model_type=model_type,
            )
            # Predict without DB transaction in batch worker (or can be configured with DB)
            resp, _ = service.predict(inp, db=None)
            results.append(
                {
                    "id": item_id,
                    "status": "success",
                    "prediction": resp.model_dump(),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": item_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    return {
        "task_id": self.request.id,
        "total_processed": len(items),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }


@celery_app.task(name="tasks.retrain_model_task", bind=True)
def retrain_model_task(
    self: Any,
    data_path: str | None = None,
    max_features: int = 5000,
) -> dict[str, Any]:
    """Execute background pipeline to retrain and evaluate the model."""
    logger.info(f"Triggering background model retraining task {self.request.id}.")
    output = run_training_pipeline(
        data_path=data_path,
        max_features=max_features,
        use_mlflow=False,
    )
    return {
        "task_id": self.request.id,
        "status": "completed",
        "output": output,
    }
