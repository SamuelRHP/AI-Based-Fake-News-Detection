"""Text preprocessing and cleaning pipeline for news articles."""

import re
import unicodedata
from typing import Any


class TextPreprocessor:
    """Preprocesses and normalizes raw article text for model ingestion and explainability."""

    def __init__(self, lowercase: bool = True, strip_html: bool = True) -> None:
        """Initialize preprocessor flags.

        Args:
            lowercase: Whether to convert text to lower case.
            strip_html: Whether to strip HTML tags from input text.
        """
        self.lowercase = lowercase
        self.strip_html = strip_html
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._html_pattern = re.compile(r"<.*?>")
        self._whitespace_pattern = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """Clean raw text string by removing noise while preserving semantic integrity.

        Args:
            text: Raw input string.

        Returns:
            Normalized and cleaned string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Normalize unicode (e.g., curly quotes, accents)
        cleaned = unicodedata.normalize("NFKD", text)

        # Remove HTML markup if requested
        if self.strip_html:
            cleaned = self._html_pattern.sub(" ", cleaned)

        # Remove raw URLs inside text
        cleaned = self._url_pattern.sub(" ", cleaned)

        # Normalize casing
        if self.lowercase:
            cleaned = cleaned.lower()

        # Condense multiple whitespaces/newlines into single space
        cleaned = self._whitespace_pattern.sub(" ", cleaned).strip()
        return cleaned

    def extract_sentences(self, text: str) -> list[str]:
        """Split article text into individual sentences for granular attribution.

        Args:
            text: Input article text.

        Returns:
            List of non-empty sentence strings.
        """
        if not text:
            return []
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw_sentences if s.strip()]

    def transform(self, texts: list[str]) -> list[str]:
        """Batch transform a list of texts.

        Args:
            texts: List of raw strings.

        Returns:
            List of cleaned strings.
        """
        return [self.clean_text(t) for t in texts]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Support scikit-learn pipeline parameter inspection."""
        return {"lowercase": self.lowercase, "strip_html": self.strip_html}
