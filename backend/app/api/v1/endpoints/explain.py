"""Explainability endpoint with token-level highlights and driver words."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.article import ArticleInput
from backend.app.schemas.explanation import ExplanationResponse
from backend.app.services.explanation_service import ExplanationService

router = APIRouter()
explanation_service = ExplanationService()


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict and explain word-level drivers",
    description="Returns verdict, confidence score, and token-level LIME/SHAP attributions with highlight weights.",
)
def explain_article(
    payload: ArticleInput,
    method: str = Query(
        default="lime",
        description="Explainability framework: 'lime' or 'shap'.",
    ),
    db: Session = Depends(get_db),
) -> ExplanationResponse:
    """Generate verdict and word-level attributions."""
    try:
        return explanation_service.explain(payload, explainer_type=method, db=db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability computation failed: {str(exc)}",
        ) from exc
