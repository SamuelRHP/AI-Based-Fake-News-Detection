"""Integration tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "active_model" in data
    assert data["version"] == "0.1.0"


def test_predict_endpoint_success():
    payload = {
        "text": "The Federal Reserve concluded its policy meeting and held rates steady.",
        "title": "Fed Holds Rates",
        "model_type": "baseline",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in ("Fake", "Real", "Uncertain")
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert 0.0 <= data["prob_fake"] <= 1.0
    assert 0.0 <= data["prob_real"] <= 1.0
    assert "threshold_applied" in data


def test_predict_endpoint_validation_error():
    # Neither text nor url provided
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422  # Pydantic validation error


def test_explain_endpoint_lime():
    payload = {
        "text": "Shocking secret cure discovered overnight by doctors and scientists.",
        "model_type": "baseline",
    }
    response = client.post("/api/v1/explain?method=lime", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "tokens" in data
    assert len(data["tokens"]) > 0
    assert "summary" in data
    assert "top_fake_drivers" in data


def test_feedback_submission_workflow():
    # 1. Run prediction first to obtain valid prediction_id
    predict_res = client.post(
        "/api/v1/predict",
        json={"text": "Astronomers discovered a new exoplanet with water vapor."},
    )
    assert predict_res.status_code == 200
    pred_data = predict_res.json()
    prediction_id = pred_data.get("prediction_id")

    if prediction_id:
        # 2. Submit feedback
        feedback_payload = {
            "prediction_id": prediction_id,
            "user_verdict": "Real",
            "comments": "Verified against Nature journal publication.",
        }
        feedback_res = client.post("/api/v1/feedback", json=feedback_payload)
        assert feedback_res.status_code == 201
        fb_data = feedback_res.json()
        assert fb_data["prediction_id"] == prediction_id
        assert fb_data["user_verdict"] == "Real"


def test_batch_predict_endpoint():
    payload = {
        "items": [
            {"id": "art-1", "text": "Federal Reserve monetary policy committee."},
            {"id": "art-2", "text": "Shocking extraterrestrial secret revealed."},
        ],
        "model_type": "baseline",
    }
    response = client.post("/api/v1/batch-predict", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["total_items"] == 2
