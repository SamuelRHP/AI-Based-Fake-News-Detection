"""Unit tests for baseline and transformer classification models."""

import pytest

from fake_news_detector.models.baseline import TfidfLogisticRegressionClassifier
from fake_news_detector.models.transformer import TransformerClassifier


def test_baseline_fit_predict_proba(synthetic_dataframe):
    model = TfidfLogisticRegressionClassifier(max_features=200)
    assert not model.is_fitted

    texts = synthetic_dataframe["text"].tolist()
    labels = synthetic_dataframe["label"].tolist()
    model.fit(texts, labels)

    assert model.is_fitted
    preds = model.predict(texts[:5])
    assert len(preds) == 5
    assert all(p in (0, 1) for p in preds)

    probs = model.predict_proba(texts[:5])
    assert probs.shape == (5, 2)
    # Check probabilities sum to 1.0
    for p in probs:
        assert pytest.approx(sum(p), rel=1e-3) == 1.0


def test_baseline_save_and_load(synthetic_dataframe, tmp_path):
    model = TfidfLogisticRegressionClassifier(max_features=200)
    texts = synthetic_dataframe["text"].tolist()
    labels = synthetic_dataframe["label"].tolist()
    model.fit(texts, labels)

    saved_file = model.save(tmp_path)
    assert saved_file.exists()

    loaded_model = TfidfLogisticRegressionClassifier().load(tmp_path)
    assert loaded_model.is_fitted

    orig_probs = model.predict_proba(texts[:3])
    loaded_probs = loaded_model.predict_proba(texts[:3])
    assert (orig_probs == loaded_probs).all()


def test_transformer_classifier_fallback():
    model = TransformerClassifier()
    sample = ["Shocking miracle conspiracy in the capital", "Regular news report"]
    probs = model.predict_proba(sample)
    assert probs.shape == (2, 2)
    assert probs[0, 1] > probs[1, 1]  # Sensational sample has higher P(Fake)
