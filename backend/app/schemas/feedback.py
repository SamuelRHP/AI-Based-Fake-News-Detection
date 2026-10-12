"""Pydantic schemas for ground-truth feedback collection."""

from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    """Payload to register human feedback or correction."""

    prediction_id: str = Field(..., description="UUID of the prediction record.")
    user_verdict: str = Field(..., description="Human assessment: 'Fake', 'Real', or 'Uncertain'.")
    comments: str | None = Field(
        default=None, description="Optional explanation or context for correction."
    )


class FeedbackResponse(BaseModel):
    """Stored feedback confirmation."""

    id: str
    prediction_id: str
    user_verdict: str
    comments: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
