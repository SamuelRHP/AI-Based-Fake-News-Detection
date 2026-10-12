"""Human feedback collection and verification endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Feedback, PredictionRecord
from backend.app.db.session import get_db
from backend.app.schemas.feedback import FeedbackCreate, FeedbackResponse

router = APIRouter()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit ground-truth user correction on a prediction",
)
def submit_feedback(
    payload: FeedbackCreate,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    """Submit human review or correction for an existing prediction."""
    # Verify prediction exists if stored
    stmt = select(PredictionRecord).where(PredictionRecord.id == payload.prediction_id)
    record = db.scalar(stmt)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record '{payload.prediction_id}' not found.",
        )

    feedback = Feedback(
        prediction_id=payload.prediction_id,
        user_verdict=payload.user_verdict,
        comments=payload.comments,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return FeedbackResponse.model_validate(feedback)


@router.get(
    "/feedback",
    response_model=list[FeedbackResponse],
    summary="List recent human feedback records",
)
def list_feedbacks(
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[FeedbackResponse]:
    """Retrieve feedback records for model retraining evaluation."""
    stmt = select(Feedback).order_by(Feedback.created_at.desc()).limit(limit)
    records = db.scalars(stmt).all()
    return [FeedbackResponse.model_validate(r) for r in records]
