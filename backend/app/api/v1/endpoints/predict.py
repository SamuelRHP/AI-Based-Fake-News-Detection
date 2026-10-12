"""Prediction endpoint for single articles."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.article import ArticleInput
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict authenticity of a news article",
    description="Takes raw text and/or URL, extracts text, runs inference, and returns Fake/Real/Uncertain verdict with confidence score.",
)
def predict_article(
    payload: ArticleInput,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Classify news article text or URL."""
    try:
        response, _ = prediction_service.predict(payload, db=db)
        return response
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        ) from exc
