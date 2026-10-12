"""Integration tests for background worker tasks."""

from backend.app.workers.tasks import predict_batch_task


def test_predict_batch_task_synchronous():
    items = [
        {"id": "doc-01", "text": "Federal Reserve economic outlook speech."},
        {"id": "doc-02", "text": "Doctors hate this one secret cure discovery."},
    ]
    # Test task execution logic directly
    output = predict_batch_task(items=items, model_type="baseline")

    assert output["total_processed"] == 2
    assert output["successful"] == 2
    assert output["failed"] == 0
    assert len(output["results"]) == 2
    first_res = output["results"][0]
    assert first_res["status"] == "success"
    assert "prediction" in first_res
    assert first_res["prediction"]["verdict"] in ("Fake", "Real", "Uncertain")

