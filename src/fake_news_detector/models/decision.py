"""Decision logic and threshold calibration for Fake/Real/Uncertain verdicts."""

from dataclasses import dataclass
from typing import Literal

from fake_news_detector.config import settings

VerdictType = Literal["Fake", "Real", "Uncertain"]


@dataclass(frozen=True)
class DecisionResult:
    """Encapsulates final verdict and calibrated confidence metrics."""

    verdict: VerdictType
    confidence_score: float
    prob_fake: float
    prob_real: float
    requires_human_review: bool
    threshold_applied: dict[str, float]


class DecisionEngine:
    """Translates model class probabilities into production verdicts and confidence bands."""

    def __init__(
        self,
        fake_threshold_high: float | None = None,
        fake_threshold_low: float | None = None,
    ) -> None:
        """Initialize decision thresholds.

        Args:
            fake_threshold_high: Minimum P(Fake) required to classify as Fake (default: 0.70).
            fake_threshold_low: Maximum P(Fake) allowed to classify as Real (default: 0.30).
        """
        self.fake_threshold_high = (
            fake_threshold_high if fake_threshold_high is not None else settings.FAKE_THRESHOLD_HIGH
        )
        self.fake_threshold_low = (
            fake_threshold_low if fake_threshold_low is not None else settings.FAKE_THRESHOLD_LOW
        )

        if self.fake_threshold_low >= self.fake_threshold_high:
            raise ValueError(
                f"Low threshold ({self.fake_threshold_low}) must be strictly less than "
                f"high threshold ({self.fake_threshold_high})."
            )

    def evaluate(self, prob_fake: float) -> DecisionResult:
        """Evaluate a single probability score.

        Confidence formula:
            confidence = 2 * abs(prob_fake - 0.5)
            Maps 0.5 (ambiguous) -> 0.0 confidence, and 0.0 or 1.0 -> 1.0 confidence.

        Args:
            prob_fake: Model probability that the text is fake [0.0, 1.0].

        Returns:
            DecisionResult dataclass.
        """
        prob_fake_clamped = max(0.0, min(1.0, float(prob_fake)))
        prob_real = 1.0 - prob_fake_clamped
        confidence_score = round(2.0 * abs(prob_fake_clamped - 0.5), 4)

        if prob_fake_clamped >= self.fake_threshold_high:
            verdict: VerdictType = "Fake"
            requires_review = False
        elif prob_fake_clamped <= self.fake_threshold_low:
            verdict = "Real"
            requires_review = False
        else:
            verdict = "Uncertain"
            requires_review = True

        return DecisionResult(
            verdict=verdict,
            confidence_score=confidence_score,
            prob_fake=round(prob_fake_clamped, 4),
            prob_real=round(prob_real, 4),
            requires_human_review=requires_review,
            threshold_applied={
                "high": self.fake_threshold_high,
                "low": self.fake_threshold_low,
            },
        )
