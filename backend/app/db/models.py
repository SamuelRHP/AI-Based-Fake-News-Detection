"""SQLAlchemy ORM models for articles, predictions, and human feedback."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


def utc_now() -> datetime:
    """Generate current UTC datetime with timezone awareness."""
    return datetime.now(UTC)


class Article(Base):
    """Stores ingested news articles from text or web scraping."""

    __tablename__ = "articles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    predictions: Mapped[list["PredictionRecord"]] = relationship(
        "PredictionRecord", back_populates="article", cascade="all, delete-orphan"
    )


class PredictionRecord(Base):
    """Historical audit log of model predictions and confidence scores."""

    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    article_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    prob_fake: Mapped[float] = mapped_column(Float, nullable=False)
    prob_real: Mapped[float] = mapped_column(Float, nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    requires_human_review: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    article: Mapped["Article"] = relationship("Article", back_populates="predictions")
    feedbacks: Mapped[list["Feedback"]] = relationship(
        "Feedback", back_populates="prediction", cascade="all, delete-orphan"
    )


class Feedback(Base):
    """User and reviewer feedback on predictions for active learning loops."""

    __tablename__ = "feedbacks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    prediction: Mapped["PredictionRecord"] = relationship(
        "PredictionRecord", back_populates="feedbacks"
    )
