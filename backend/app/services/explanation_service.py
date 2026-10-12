"""Service generating local token attributions and UI highlights."""

from sqlalchemy.orm import Session

from backend.app.schemas.article import ArticleInput
from backend.app.schemas.explanation import (
    DriverWord,
    ExplanationResponse,
    TokenAttribution,
)
from backend.app.services.prediction_service import PredictionService
from fake_news_detector.explainability.formatter import ExplanationFormatter
from fake_news_detector.explainability.lime_explainer import LimeExplainerWrapper
from fake_news_detector.explainability.shap_explainer import ShapExplainerWrapper
from fake_news_detector.models.registry import ModelRegistry


class ExplanationService:
    """Generates combined prediction and token-level explanations."""

    def __init__(self) -> None:
        self.prediction_service = PredictionService()
        self.lime_explainer = LimeExplainerWrapper()
        self.shap_explainer = ShapExplainerWrapper()

    def explain(
        self,
        article_input: ArticleInput,
        explainer_type: str = "lime",
        db: Session | None = None,
    ) -> ExplanationResponse:
        """Run prediction and compute token attribution heatmap."""
        # 1. Run prediction
        prediction_resp, article_text = self.prediction_service.predict(article_input, db=db)

        # 2. Get active classifier predict callable
        model = ModelRegistry.get_model(model_type=prediction_resp.model_type)

        # 3. Compute attributions
        if explainer_type.lower() == "shap":
            explainer_res = self.shap_explainer.explain_instance(
                article_text, model.predict_proba, num_features=10
            )
        else:
            explainer_res = self.lime_explainer.explain_instance(
                article_text, model.predict_proba, num_features=10
            )

        attributions = explainer_res.get("attributions", [])
        method_used = explainer_res.get("method", explainer_type)

        # 4. Format payload for UI
        formatted = ExplanationFormatter.format_explanation(
            text=article_text,
            attributions=attributions,
            verdict=prediction_resp.verdict,
        )

        tokens = [
            TokenAttribution(
                text=t["text"],
                weight=t["weight"],
                impact=t["impact"],
                intensity=t["intensity"],
            )
            for t in formatted["tokens"]
        ]

        top_fake = [
            DriverWord(word=d["word"], weight=d["weight"]) for d in formatted["top_fake_drivers"]
        ]
        top_real = [
            DriverWord(word=d["word"], weight=d["weight"]) for d in formatted["top_real_drivers"]
        ]

        return ExplanationResponse(
            prediction=prediction_resp,
            explainer_method=method_used,
            tokens=tokens,
            top_fake_drivers=top_fake,
            top_real_drivers=top_real,
            summary=formatted["summary"],
        )
