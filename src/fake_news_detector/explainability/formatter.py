"""Payload formatter for explainability visualization in web frontends."""

import re
from typing import Any, Literal

ImpactType = Literal["fake", "real", "neutral"]


class ExplanationFormatter:
    """Formats raw feature attributions into token highlights and driver word summaries."""

    @staticmethod
    def format_explanation(
        text: str,
        attributions: list[tuple[str, float]],
        verdict: str,
    ) -> dict[str, Any]:
        """Format raw attributions into a structured token-level payload.

        Args:
            text: Original article text.
            attributions: List of (word/phrase, attribution_weight) pairs.
            verdict: The predicted verdict ('Fake', 'Real', or 'Uncertain').

        Returns:
            Dictionary with token highlights, driver words, and explanatory summary.
        """
        # Build dictionary of word weights (normalized lowercase)
        weight_map = {word.lower().strip(): float(weight) for word, weight in attributions}

        # Tokenize text while preserving words and whitespace
        tokens_raw = re.findall(r"\w+|[^\w\s]|\s+", text)
        max_abs_weight = max([abs(w) for w in weight_map.values()], default=1.0) or 1.0

        formatted_tokens = []
        for tok in tokens_raw:
            cleaned = tok.lower().strip()
            weight = weight_map.get(cleaned, 0.0)

            # Determine impact classification
            if weight > 0.02:
                impact: ImpactType = "fake"
            elif weight < -0.02:
                impact = "real"
            else:
                impact = "neutral"

            # Normalize intensity between 0.0 and 1.0 for CSS opacity
            intensity = (
                min(1.0, round(abs(weight) / max_abs_weight, 3)) if impact != "neutral" else 0.0
            )

            formatted_tokens.append(
                {
                    "text": tok,
                    "weight": round(weight, 4),
                    "impact": impact,
                    "intensity": intensity,
                }
            )

        # Separate top drivers
        top_fake = [{"word": w, "weight": round(wt, 4)} for w, wt in attributions if wt > 0.0][:5]

        top_real = [{"word": w, "weight": round(abs(wt), 4)} for w, wt in attributions if wt < 0.0][
            :5
        ]

        # Natural language summary
        if verdict == "Fake" and top_fake:
            fake_words_str = ", ".join([f"'{d['word']}'" for d in top_fake[:3]])
            summary = (
                f"Prediction of Fake was primarily driven by sensational or suspicious language "
                f"such as {fake_words_str}."
            )
        elif verdict == "Real" and top_real:
            real_words_str = ", ".join([f"'{d['word']}'" for d in top_real[:3]])
            summary = (
                f"Prediction of Real was supported by objective or verifiable terminology "
                f"such as {real_words_str}."
            )
        else:
            summary = (
                "Article contains mixed or neutral stylistic markers. "
                "Confidence is moderate or uncertain; manual review is advised."
            )

        return {
            "tokens": formatted_tokens,
            "top_fake_drivers": top_fake,
            "top_real_drivers": top_real,
            "summary": summary,
        }
