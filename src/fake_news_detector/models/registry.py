"""Model registry for managing, caching, and serving model versions."""

import logging
from pathlib import Path

from fake_news_detector.config import settings
from fake_news_detector.data.loader import create_synthetic_news_dataset
from fake_news_detector.models.base import BaseFakeNewsClassifier
from fake_news_detector.models.baseline import TfidfLogisticRegressionClassifier
from fake_news_detector.models.transformer import TransformerClassifier

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry and cache for trained model instances."""

    _active_model: BaseFakeNewsClassifier | None = None
    _active_model_type: str | None = None

    @classmethod
    def get_model(
        cls,
        model_type: str | None = None,
        artifacts_dir: Path | None = None,
        auto_fit_if_missing: bool = True,
    ) -> BaseFakeNewsClassifier:
        """Retrieve active model singleton, or instantiate and load from artifacts directory.

        Args:
            model_type: 'baseline' or 'transformer' (defaults to settings.DEFAULT_MODEL_TYPE).
            artifacts_dir: Path where model weights are stored.
            auto_fit_if_missing: If True and model weights are not found, trains a minimal
                                baseline on synthetic data to guarantee a runnable system.

        Returns:
            Fitted BaseFakeNewsClassifier instance.
        """
        target_type = model_type or settings.DEFAULT_MODEL_TYPE
        target_dir = artifacts_dir or settings.ARTIFACTS_DIR

        if cls._active_model is not None and cls._active_model_type == target_type:
            return cls._active_model

        if target_type == "transformer":
            model: BaseFakeNewsClassifier = TransformerClassifier()
            model_path = target_dir / "transformer"
            if model_path.exists():
                model.load(model_path)
            else:
                model.fit(["dummy"], [0])
        else:
            model = TfidfLogisticRegressionClassifier()
            model_file = target_dir / "baseline_model.joblib"
            if model_file.exists():
                model.load(target_dir)
            elif auto_fit_if_missing:
                logger.info("No saved baseline model found. Training on seed dataset...")
                df = create_synthetic_news_dataset(num_samples=20)
                model.fit(df["text"].tolist(), df["label"].tolist())
                try:
                    model.save(target_dir)
                except Exception as exc:
                    logger.warning(f"Could not persist seed model: {exc}")

        cls._active_model = model
        cls._active_model_type = target_type
        return model

    @classmethod
    def reset(cls) -> None:
        """Reset cached singleton instance."""
        cls._active_model = None
        cls._active_model_type = None
