"""Unit tests for explainers and payload formatting."""

from fake_news_detector.explainability.formatter import ExplanationFormatter
from fake_news_detector.explainability.lime_explainer import LimeExplainerWrapper
from fake_news_detector.explainability.shap_explainer import ShapExplainerWrapper


def test_lime_explainer_fallback(trained_baseline_model):
    explainer = LimeExplainerWrapper()
    text = "Shocking miracle discovery announced overnight by secret scientists."
    res = explainer.explain_instance(text, trained_baseline_model.predict_proba, num_features=5)

    assert "attributions" in res
    assert len(res["attributions"]) > 0
    assert any(
        isinstance(pair[0], str) and isinstance(pair[1], float) for pair in res["attributions"]
    )


def test_shap_explainer_fallback(trained_baseline_model):
    explainer = ShapExplainerWrapper()
    text = "Federal Reserve announces interest rate policy update."
    res = explainer.explain_instance(text, trained_baseline_model.predict_proba, num_features=5)

    assert "attributions" in res
    assert len(res["attributions"]) > 0


def test_explanation_formatter():
    text = "Secret cure discovered by doctors."
    attributions = [("secret", 0.45), ("cure", 0.30), ("doctors", -0.10)]
    formatted = ExplanationFormatter.format_explanation(text, attributions, verdict="Fake")

    assert "tokens" in formatted
    assert "top_fake_drivers" in formatted
    assert "top_real_drivers" in formatted
    assert "summary" in formatted
    assert len(formatted["top_fake_drivers"]) == 2
    assert len(formatted["top_real_drivers"]) == 1

    # Check token highlights
    token_texts = [t["text"].lower() for t in formatted["tokens"]]
    assert "secret" in token_texts
    secret_token = next(t for t in formatted["tokens"] if t["text"].lower() == "secret")
    assert secret_token["impact"] == "fake"
    assert secret_token["intensity"] > 0
