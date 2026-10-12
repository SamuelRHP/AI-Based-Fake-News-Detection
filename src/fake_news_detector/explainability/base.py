"""Abstract base interface for model explainability."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


class BaseExplainer(ABC):
    """Abstract interface for local interpretability and token attribution."""

    @abstractmethod
    def explain_instance(
        self,
        text: str,
        predict_fn: Callable[[list[str]], np.ndarray],
        num_features: int = 10,
    ) -> dict[str, Any]:
        """Generate word- and token-level attribution scores for a single article text.

        Args:
            text: Raw input article string.
            predict_fn: Callable returning class probability matrix [P(Real), P(Fake)].
            num_features: Maximum number of top explanatory features to return.

        Returns:
            Dictionary containing feature attribution pairs and metadata.
        """
        pass
