"""SHAP-based token and feature attribution."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from fake_news_detector.explainability.base import BaseExplainer

logger = logging.getLogger(__name__)


class ShapExplainerWrapper(BaseExplainer):
    """Computes Shapley values for input text tokens using SHAP."""

    def __init__(self) -> None:
        """Initialize SHAP wrapper."""
        self._explainer: Any = None

    def explain_instance(
        self,
        text: str,
        predict_fn: Callable[[list[str]], np.ndarray],
        num_features: int = 10,
    ) -> dict[str, Any]:
        """Compute token-level SHAP values for the given text.

        Returns:
            Dictionary with feature attributions (word, weight).
        """
        if not text.strip():
            return {"attributions": [], "explainer_type": "shap", "method": "fallback"}

        try:
            import shap

            # Use Explainer with custom predict callable
            explainer = shap.Explainer(predict_fn, masker=shap.maskers.Text())
            shap_values = explainer([text])
            # Extract word values for class 1 (Fake)
            words = shap_values.data[0]
            values = shap_values.values[0, :, 1]
            attributions = [
                (str(w), float(v)) for w, v in zip(words, values, strict=False) if abs(v) > 0.001
            ]
            attributions.sort(key=lambda x: abs(x[1]), reverse=True)
            return {
                "attributions": attributions[:num_features],
                "explainer_type": "shap",
                "method": "shap_partition",
            }
        except Exception as exc:
            logger.debug(f"SHAP full explainer bypassed or failed: {exc}")

        # Exact Leave-One-Covariate-Out approximation fallback
        words = list(dict.fromkeys(text.split()))[:25]
        base_proba = predict_fn([text])[0][1]
        attributions = []

        for word in words:
            if len(word) < 3:
                continue
            perturbed = text.replace(word, "")
            perturbed_proba = predict_fn([perturbed])[0][1]
            shap_approx = float(base_proba - perturbed_proba)
            if abs(shap_approx) > 0.001:
                attributions.append((word, shap_approx))

        attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return {
            "attributions": attributions[:num_features],
            "explainer_type": "shap",
            "method": "shap_loco_approximation",
        }
