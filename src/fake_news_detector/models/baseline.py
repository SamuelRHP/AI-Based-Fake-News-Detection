"""Baseline classification model combining TF-IDF and Logistic Regression."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fake_news_detector.models.base import BaseFakeNewsClassifier


class TfidfLogisticRegressionClassifier(BaseFakeNewsClassifier):
    """Scikit-learn baseline model using TF-IDF n-grams with regularized Logistic Regression."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        c_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """Initialize baseline hyperparameters."""
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.c_regularization = c_regularization
        self.random_state = random_state
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                        sublinear_tf=True,
                        stop_words="english",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=self.c_regularization,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        return "tfidf-logistic-regression"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def fit(self, texts: list[str], labels: list[int]) -> "TfidfLogisticRegressionClassifier":
        """Fit the TF-IDF vectorizer and train the Logistic Regression classifier."""
        if not texts or len(texts) != len(labels):
            raise ValueError("Input texts and labels must be non-empty and equal length.")

        self._pipeline.fit(texts, labels)
        self._is_fitted = True
        return self

    def predict(self, texts: list[str]) -> list[int]:
        """Predict binary classification labels."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() before predict().")
        preds: np.ndarray = self._pipeline.predict(texts)
        return [int(p) for p in preds]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities: [P(Real), P(Fake)]."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() before predict_proba().")
        probs: np.ndarray = self._pipeline.predict_proba(texts)
        return probs

    def save(self, output_dir: str | Path) -> Path:
        """Save pipeline and metadata to target directory."""
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        model_file = target_dir / "baseline_model.joblib"
        metadata_file = target_dir / "metadata.json"

        joblib.dump(self._pipeline, model_file)

        metadata = {
            "model_name": self.model_name,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "c_regularization": self.c_regularization,
            "random_state": self.random_state,
            "is_fitted": self._is_fitted,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return model_file

    def load(self, model_dir: str | Path) -> "TfidfLogisticRegressionClassifier":
        """Load fitted pipeline and hyperparameter metadata from directory."""
        target_dir = Path(model_dir)
        model_file = target_dir / "baseline_model.joblib"
        metadata_file = target_dir / "metadata.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at: {model_file}")

        self._pipeline = joblib.load(model_file)
        self._is_fitted = True

        if metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as f:
                metadata = json.load(f)
                self.max_features = metadata.get("max_features", self.max_features)
                self.c_regularization = metadata.get("c_regularization", self.c_regularization)

        return self
