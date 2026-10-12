"""Training pipeline for Fake News Detection with MLflow tracking."""

import logging
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from fake_news_detector.config import settings
from fake_news_detector.data.loader import DataLoader, create_synthetic_news_dataset
from fake_news_detector.data.preprocessor import TextPreprocessor
from fake_news_detector.evaluation.metrics import ModelEvaluator
from fake_news_detector.models.baseline import TfidfLogisticRegressionClassifier

logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_features: int = 5000,
    c_regularization: float = 1.0,
    test_size: float = 0.2,
    use_mlflow: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end data loading, preprocessing, model training, and evaluation.

    Args:
        data_path: Optional path to training CSV dataset.
                   If not provided, uses synthetic seed data.
        output_dir: Directory where trained artifacts are saved.
        max_features: Maximum TF-IDF vocabulary size.
        c_regularization: Logistic regression penalty strength.
        test_size: Holdout validation split fraction.
        use_mlflow: Whether to log run parameters and metrics to MLflow.

    Returns:
        Dictionary containing evaluation metrics and saved model path.
    """
    out_dir = Path(output_dir or settings.ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest dataset
    if data_path and Path(data_path).exists():
        logger.info(f"Loading dataset from {data_path}")
        df = DataLoader.load_from_csv(data_path)
    else:
        logger.info("Using seed synthetic dataset for pipeline execution.")
        df = create_synthetic_news_dataset(num_samples=30)

    # 2. Preprocess text
    preprocessor = TextPreprocessor()
    df["cleaned_text"] = preprocessor.transform(df["text"].tolist())

    # 3. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        df["cleaned_text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=settings.RANDOM_STATE,
        stratify=df["label"].tolist(),
    )

    # 4. Instantiate and fit model
    model = TfidfLogisticRegressionClassifier(
        max_features=max_features,
        c_regularization=c_regularization,
        random_state=settings.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    metrics = ModelEvaluator.calculate_metrics(y_val, y_pred, y_prob)

    # 6. Save artifact
    saved_path = model.save(out_dir)

    # 7. Optional MLflow tracking
    if use_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="baseline-tfidf-logreg"):
                mlflow.log_params(
                    {
                        "model_type": "baseline",
                        "max_features": max_features,
                        "c_regularization": c_regularization,
                        "test_size": test_size,
                    }
                )
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_artifact(str(saved_path))
        except Exception as exc:
            logger.warning(f"MLflow logging bypassed: {exc}")

    return {
        "model_path": str(saved_path),
        "metrics": metrics,
        "sample_count_train": len(X_train),
        "sample_count_val": len(X_val),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_training_pipeline()
    print("Training Pipeline Execution Finished:")
    print(results)
