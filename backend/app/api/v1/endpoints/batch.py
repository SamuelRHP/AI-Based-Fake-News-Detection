"""Asynchronous batch prediction endpoints using Celery."""

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.prediction import BatchPredictionRequest, BatchTaskResponse
from backend.app.workers.celery_app import celery_app
from backend.app.workers.tasks import predict_batch_task

router = APIRouter()


@router.post(
    "/batch-predict",
    response_model=BatchTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue asynchronous batch prediction job",
)
def submit_batch_predictions(payload: BatchPredictionRequest) -> BatchTaskResponse:
    """Submit a batch of articles to the background Celery queue."""
    if not payload.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch payload cannot be empty.",
        )

    items_dicts = [item.model_dump() for item in payload.items]

    try:
        # Enqueue Celery task asynchronously
        task = predict_batch_task.delay(items=items_dicts, model_type=payload.model_type)
        return BatchTaskResponse(
            task_id=task.id,
            status="QUEUED",
            total_items=len(payload.items),
            message="Batch job accepted and queued for worker processing.",
        )
    except Exception:
        # Fallback for environments where Redis/Celery worker is offline during local test
        task_id = "local-sync-task"
        res = predict_batch_task(items=items_dicts, model_type=payload.model_type)
        return BatchTaskResponse(
            task_id=task_id,
            status="COMPLETED",
            total_items=len(payload.items),
            message=f"Executed synchronously (Worker offline): {res['successful']} succeeded.",
        )


@router.get(
    "/batch-predict/{task_id}",
    summary="Query status of an asynchronous batch task",
)
def get_batch_task_status(task_id: str) -> dict:
    """Fetch status and result of a queued batch prediction job."""
    task_result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task_result.status,
        "ready": task_result.ready(),
        "result": task_result.result if task_result.ready() else None,
    }
