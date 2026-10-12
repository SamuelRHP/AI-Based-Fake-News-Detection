"""Abstract base interface for fake news classification models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseFakeNewsClassifier(ABC):
    """Abstract contract that all fake news detection models must implement."""

    @abstractmethod
    def fit(self, texts: list[str], labels: list[int]) -> "BaseFakeNewsClassifier":
        """Train the model on texts and binary ground-truth labels (1=Fake, 0=Real)."""
        pass

    @abstractmethod
    def predict(self, texts: list[str]) -> list[int]:
        """Predict binary class labels (1=Fake, 0=Real)."""
        pass

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probability distribution of shape (n_samples, 2).

        Column 0 represents P(Real), Column 1 represents P(Fake).
        """
        pass

    @abstractmethod
    def save(self, output_dir: str | Path) -> Path:
        """Persist model weights and metadata to disk."""
        pass

    @abstractmethod
    def load(self, model_dir: str | Path) -> "BaseFakeNewsClassifier":
        """Load model weights and metadata from disk."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the identifier/name of the model architecture."""
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if model is trained and ready for inference."""
        pass
