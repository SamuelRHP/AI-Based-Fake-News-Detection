"""Pydantic schemas for explainability and token attributions."""

from pydantic import BaseModel, Field

from backend.app.schemas.prediction import PredictionResponse
from fake_news_detector.explainability.formatter import ImpactType


class TokenAttribution(BaseModel):
    """Token representation with explanatory attribution weight."""

    text: str
    weight: float
    impact: ImpactType = Field(
        ..., description="'fake' (pushes to fake), 'real' (pushes to real), or 'neutral'"
    )
    intensity: float = Field(
        ..., description="Normalized attribution intensity [0.0 - 1.0] for UI highlight opacity"
    )


class DriverWord(BaseModel):
    """Word or n-gram with substantial contribution to the verdict."""

    word: str
    weight: float


class ExplanationResponse(BaseModel):
    """Combined prediction and word-level explanation payload."""

    prediction: PredictionResponse
    explainer_method: str
    tokens: list[TokenAttribution]
    top_fake_drivers: list[DriverWord]
    top_real_drivers: list[DriverWord]
    summary: str
