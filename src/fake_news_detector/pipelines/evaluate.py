"""Evaluation pipeline for scoring trained models against evaluation sets."""

import json
import logging
from pathlib import Path
from typing import Any

from fake_news_detector.config import settings
from fake_news_detector.data.loader import DataLoader, create_synthetic_news_dataset
from fake_news_detector.data.preprocessor import TextPreprocessor
from fake_news_detector.evaluation.metrics import ModelEvaluator
from fake_news_detector.models.baseline import TfidfLogisticRegressionClassifier

logger = logging.getLogger(__name__)


def run_evaluation_pipeline(
    model_dir: str | Path | None = None,
    eval_data_path: str | Path | None = None,
    output_metrics_file: str | Path | None = None,
) -> dict[str, Any]:
    """Load model from disk and compute evaluation metrics against holdout data."""
    target_model_dir = Path(model_dir or settings.ARTIFACTS_DIR)
    model = TfidfLogisticRegressionClassifier().load(target_model_dir)

    if eval_data_path and Path(eval_data_path).exists():
        df = DataLoader.load_from_csv(eval_data_path)
    else:
        df = create_synthetic_news_dataset(num_samples=20)

    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.transform(df["text"].tolist())
    labels = df["label"].tolist()

    y_pred = model.predict(cleaned_texts)
    y_prob = model.predict_proba(cleaned_texts)

    metrics = ModelEvaluator.calculate_metrics(labels, y_pred, y_prob)

    if output_metrics_file:
        out_path = Path(output_metrics_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_metrics = run_evaluation_pipeline()
    print("Evaluation Pipeline Results:")
    print(eval_metrics)
