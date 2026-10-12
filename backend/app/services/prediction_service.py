"""Service encapsulating article extraction, preprocessing, and model scoring."""

import logging

from sqlalchemy.orm import Session

from backend.app.core.config import backend_settings
from backend.app.db.models import Article, PredictionRecord
from backend.app.schemas.article import ArticleInput
from backend.app.schemas.prediction import PredictionResponse
from fake_news_detector.data.preprocessor import TextPreprocessor
from fake_news_detector.data.scraper import ArticleScraper
from fake_news_detector.models.decision import DecisionEngine
from fake_news_detector.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class PredictionService:
    """Orchestrates end-to-end news verification workflow."""

    def __init__(self) -> None:
        self.scraper = ArticleScraper()
        self.preprocessor = TextPreprocessor()
        self.decision_engine = DecisionEngine()

    def predict(
        self,
        article_input: ArticleInput,
        db: Session | None = None,
    ) -> tuple[PredictionResponse, str]:
        """Perform classification on submitted text or URL.

        Args:
            article_input: User input containing text and/or URL.
            db: Optional database session for audit logging.

        Returns:
            Tuple of (PredictionResponse, original_or_scraped_text).
        """
        raw_text = article_input.text or ""
        title = article_input.title or ""
        url = article_input.url

        # Scrape article if URL provided and text is empty
        if url and not raw_text.strip():
            logger.info(f"Extracting article from URL: {url}")
            scraped = self.scraper.fetch_and_extract(url)
            if not scraped["success"]:
                raise ValueError(scraped["error"] or f"Could not scrape article from {url}")
            raw_text = scraped["text"]
            title = title or scraped["title"]

        if not raw_text.strip():
            raise ValueError("No valid article text available for evaluation.")

        # Clean text
        cleaned_text = self.preprocessor.clean_text(raw_text)

        # Select model and infer
        model_type = article_input.model_type or backend_settings.DEFAULT_MODEL_TYPE
        model = ModelRegistry.get_model(model_type=model_type)
        probs = model.predict_proba([cleaned_text])[0]
        prob_fake = float(probs[1])

        # Evaluate decision bands
        decision = self.decision_engine.evaluate(prob_fake)

        article_id = None
        prediction_id = None

        # Persist audit record if database session is provided
        if db is not None:
            try:
                article = Article(
                    url=url,
                    title=title,
                    raw_text=raw_text,
                    cleaned_text=cleaned_text,
                )
                db.add(article)
                db.flush()

                record = PredictionRecord(
                    article_id=article.id,
                    verdict=decision.verdict,
                    confidence_score=decision.confidence_score,
                    prob_fake=decision.prob_fake,
                    prob_real=decision.prob_real,
                    model_type=model_type,
                    requires_human_review=decision.requires_human_review,
                )
                db.add(record)
                db.commit()
                db.refresh(record)

                article_id = article.id
                prediction_id = record.id
            except Exception as exc:
                db.rollback()
                logger.error(f"Failed to persist prediction audit log: {exc}")

        response = PredictionResponse(
            prediction_id=prediction_id,
            article_id=article_id,
            title=title if title else None,
            verdict=decision.verdict,
            confidence_score=decision.confidence_score,
            prob_fake=decision.prob_fake,
            prob_real=decision.prob_real,
            requires_human_review=decision.requires_human_review,
            model_type=model_type,
            threshold_applied=decision.threshold_applied,
        )
        return response, raw_text
