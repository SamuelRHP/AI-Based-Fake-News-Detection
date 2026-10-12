"""Shared pytest fixtures."""

import pytest

from fake_news_detector.data.loader import create_synthetic_news_dataset
from fake_news_detector.models.baseline import TfidfLogisticRegressionClassifier
from fake_news_detector.models.decision import DecisionEngine


@pytest.fixture
def synthetic_dataframe():
    """Provide deterministic test dataset."""
    return create_synthetic_news_dataset(num_samples=20)


@pytest.fixture
def trained_baseline_model(synthetic_dataframe):
    """Provide a trained in-memory baseline classifier."""
    model = TfidfLogisticRegressionClassifier(max_features=500)
    model.fit(
        synthetic_dataframe["text"].tolist(),
        synthetic_dataframe["label"].tolist(),
    )
    return model


@pytest.fixture
def decision_engine():
    """Provide standard decision engine."""
    return DecisionEngine(fake_threshold_high=0.70, fake_threshold_low=0.30)
