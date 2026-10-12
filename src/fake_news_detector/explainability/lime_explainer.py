"""LIME-based local interpretable model-agnostic explanations."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from fake_news_detector.explainability.base import BaseExplainer

logger = logging.getLogger(__name__)


class LimeExplainerWrapper(BaseExplainer):
    """Generates local feature attributions using LIME (Local Interpretable Model-agnostic Explanations)."""

    def __init__(self, class_names: list[str] | None = None, random_state: int = 42) -> None:
        """Initialize LIME wrapper with class names and random seed."""
        self.class_names = class_names or ["Real", "Fake"]
        self.random_state = random_state
        self._explainer: Any = None

    @property
    def explainer(self) -> Any:
        """Lazy load LimeTextExplainer."""
        if self._explainer is None:
            try:
                from lime.lime_text import LimeTextExplainer

                self._explainer = LimeTextExplainer(
                    class_names=self.class_names,
                    random_state=self.random_state,
                    bow=False,
                )
            except ImportError:
                self._explainer = None
        return self._explainer

    def explain_instance(
        self,
        text: str,
        predict_fn: Callable[[list[str]], np.ndarray],
        num_features: int = 10,
    ) -> dict[str, Any]:
        """Explain the prediction of a single news article.

        Positive weights indicate words driving toward class 1 (Fake),
        while negative weights indicate words driving toward class 0 (Real).
        """
        if not text.strip():
            return {"attributions": [], "explainer_type": "lime", "method": "fallback"}

        if self.explainer is not None:
            try:
                exp = self.explainer.explain_instance(
                    text_instance=text,
                    classifier_fn=predict_fn,
                    num_features=num_features,
                    labels=[1],
                )
                attributions = exp.as_list(label=1)
                return {
                    "attributions": attributions,
                    "explainer_type": "lime",
                    "method": "lime_text",
                }
            except Exception as exc:
                logger.warning(f"LIME calculation failed, falling back to perturbation: {exc}")

        # Local perturbation fallback when LIME library is unavailable
        words = list(dict.fromkeys(text.split()))[:30]
        base_proba = predict_fn([text])[0][1]
        attributions = []

        for word in words:
            if len(word) < 3:
                continue
            # Remove word to measure shift in P(Fake)
            perturbed_text = text.replace(word, "")
            perturbed_proba = predict_fn([perturbed_text])[0][1]
            diff = base_proba - perturbed_proba
            if abs(diff) > 0.001:
                attributions.append((word, float(diff)))

        attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return {
            "attributions": attributions[:num_features],
            "explainer_type": "lime",
            "method": "local_perturbation_fallback",
        }
