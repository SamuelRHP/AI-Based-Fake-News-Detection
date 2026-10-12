"""HuggingFace Transformer tokenization wrapper."""

from typing import Any


class HuggingFaceTokenizerWrapper:
    """Encapsulates Hugging Face tokenizers for transformer models."""

    def __init__(
        self, pretrained_model_name: str = "distilbert-base-uncased", max_length: int = 512
    ) -> None:
        """Initialize tokenizer configuration.

        Args:
            pretrained_model_name: Hugging Face hub model identifier.
            max_length: Maximum sequence token truncation limit.
        """
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length
        self._tokenizer: Any = None

    @property
    def tokenizer(self) -> Any:
        """Lazy load tokenizer on first access."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
            except ImportError:
                # Stub fallback for lightweight dev/test environments
                self._tokenizer = None
        return self._tokenizer

    def tokenize(self, texts: list[str]) -> dict[str, Any]:
        """Tokenize a list of texts into PyTorch/NumPy tensors or mock tokens."""
        if self.tokenizer is not None:
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        # Fallback dictionary for testing when transformers weights are not downloaded
        return {
            "input_ids": [[hash(w) % 1000 for w in t.split()[: self.max_length]] for t in texts],
            "attention_mask": [[1] * min(len(t.split()), self.max_length) for t in texts],
        }
