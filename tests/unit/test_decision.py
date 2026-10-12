"""Unit tests for the calibrated DecisionEngine."""

import pytest

from fake_news_detector.models.decision import DecisionEngine


def test_decision_engine_fake_verdict(decision_engine):
    res = decision_engine.evaluate(prob_fake=0.85)
    assert res.verdict == "Fake"
    assert res.requires_human_review is False
    assert res.confidence_score == 0.70  # 2 * abs(0.85 - 0.5)


def test_decision_engine_real_verdict(decision_engine):
    res = decision_engine.evaluate(prob_fake=0.15)
    assert res.verdict == "Real"
    assert res.requires_human_review is False
    assert res.confidence_score == 0.70  # 2 * abs(0.15 - 0.5)


def test_decision_engine_uncertain_verdict(decision_engine):
    res = decision_engine.evaluate(prob_fake=0.52)
    assert res.verdict == "Uncertain"
    assert res.requires_human_review is True
    assert res.confidence_score == 0.04  # 2 * abs(0.52 - 0.5)


def test_decision_engine_invalid_thresholds():
    with pytest.raises(ValueError):
        DecisionEngine(fake_threshold_high=0.40, fake_threshold_low=0.60)
