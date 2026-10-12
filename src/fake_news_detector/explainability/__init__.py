"""Explainability module for local and global model interpretability."""

from fake_news_detector.explainability.base import BaseExplainer
from fake_news_detector.explainability.formatter import ExplanationFormatter, ImpactType
from fake_news_detector.explainability.lime_explainer import LimeExplainerWrapper
from fake_news_detector.explainability.shap_explainer import ShapExplainerWrapper

__all__ = [
    "BaseExplainer",
    "LimeExplainerWrapper",
    "ShapExplainerWrapper",
    "ExplanationFormatter",
    "ImpactType",
]
