"""Text feature extraction using TF-IDF and surface-level linguistic signals."""

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor:
    """Combines TF-IDF n-gram vectorization with lexical and stylistic text statistics."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        """Initialize feature extractor with TF-IDF parameters.

        Args:
            max_features: Maximum vocabulary size for TF-IDF.
            ngram_range: Range of n-values for n-grams.
            sublinear_tf: Apply sublinear scaling (1 + log(tf)).
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
        )

    def fit(self, texts: list[str]) -> "TextFeatureExtractor":
        """Fit TF-IDF vocabulary on a collection of documents."""
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> Any:
        """Transform documents to TF-IDF sparse matrix."""
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]) -> Any:
        """Fit and transform documents in one pass."""
        return self.vectorizer.fit_transform(texts)

    @staticmethod
    def extract_lexical_statistics(text: str) -> dict[str, float]:
        """Compute surface-level stylistic markers often correlated with sensationalism.

        Args:
            text: Raw input article text.

        Returns:
            Dictionary of linguistic ratios and counts.
        """
        if not text:
            return {
                "char_count": 0.0,
                "word_count": 0.0,
                "uppercase_ratio": 0.0,
                "exclamation_count": 0.0,
                "question_count": 0.0,
                "avg_word_length": 0.0,
            }

        chars = len(text)
        words = text.split()
        word_count = len(words)
        uppercase_chars = sum(1 for c in text if c.isupper())
        exclamation_count = float(text.count("!"))
        question_count = float(text.count("?"))

        uppercase_ratio = uppercase_chars / chars if chars > 0 else 0.0
        avg_word_length = float(np.mean([len(w) for w in words])) if word_count > 0 else 0.0

        return {
            "char_count": float(chars),
            "word_count": float(word_count),
            "uppercase_ratio": float(uppercase_ratio),
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "avg_word_length": avg_word_length,
        }
