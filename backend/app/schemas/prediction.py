"""Pydantic schemas for predictions and batch operations."""

from pydantic import BaseModel, Field

from fake_news_detector.models.decision import VerdictType


class PredictionResponse(BaseModel):
    """Structured result returned by the fake news prediction service."""

    prediction_id: str | None = None
    article_id: str | None = None
    title: str | None = None
    verdict: VerdictType = Field(
        ..., description="Categorical prediction: 'Fake', 'Real', or 'Uncertain'."
    )
    confidence_score: float = Field(
        ..., description="Calibrated confidence [0.0 = completely ambiguous, 1.0 = certain]."
    )
    prob_fake: float = Field(..., description="Estimated probability of being fake [0.0 - 1.0].")
    prob_real: float = Field(..., description="Estimated probability of being real [0.0 - 1.0].")
    requires_human_review: bool = Field(
        ..., description="True if verdict falls into the Uncertain confidence band."
    )
    model_type: str
    threshold_applied: dict[str, float]


class BatchPredictionItem(BaseModel):
    """Single item in a batch prediction request."""

    id: str | None = None
    text: str | None = None
    url: str | None = None
    title: str | None = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction payload."""

    items: list[BatchPredictionItem]
    model_type: str | None = None


class BatchTaskResponse(BaseModel):
    """Asynchronous background task status descriptor."""

    task_id: str
    status: str
    total_items: int
    message: str
