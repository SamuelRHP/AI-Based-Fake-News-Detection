"""Training pipeline for Fake News Detection with MLflow tracking."""

import argparse
import json
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
    ngram_range: tuple[int, int] = (1, 2),
    val_output_path: str | Path | None = None,
    metrics_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute end-to-end data loading, preprocessing, model training, and evaluation.

    Args:
        data_path: Path to training CSV (columns: title, text, label). If None, falls back to
                   a tiny synthetic dataset (tests/CI only). If a path is given but missing,
                   raises FileNotFoundError instead of silently training on fake data.
        output_dir: Directory where trained artifacts are saved.
        max_features: Maximum TF-IDF vocabulary size.
        c_regularization: Logistic regression penalty strength.
        test_size: Holdout validation split fraction.
        use_mlflow: Whether to log run parameters and metrics to MLflow.
        ngram_range: TF-IDF n-gram range.
        val_output_path: If set, the held-out validation rows are written here (raw text),
                         so evaluate.py can score the model on data it never trained on.
        metrics_output_path: If set, validation metrics are written here as JSON.

    Returns:
        Dictionary containing evaluation metrics and saved model path.
    """
    out_dir = Path(output_dir or settings.ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest dataset
    if data_path is not None:
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Training dataset not found: {data_path}")
        logger.info(f"Loading dataset from {data_path}")
        df = DataLoader.load_from_csv(data_path)
    else:
        logger.warning("No data_path given: using SYNTHETIC data (tests/CI only, not real news).")
        df = create_synthetic_news_dataset(num_samples=30)

    # 2. Preprocess text
    preprocessor = TextPreprocessor()
    df["cleaned_text"] = preprocessor.transform(df["text"].tolist())

    # 3. Train/validation split (keep raw rows so the holdout can be re-scored later)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=settings.RANDOM_STATE,
        stratify=df["label"],
    )
    X_train, y_train = train_df["cleaned_text"].tolist(), train_df["label"].tolist()
    X_val, y_val = val_df["cleaned_text"].tolist(), val_df["label"].tolist()

    # 4. Instantiate and fit model
    model = TfidfLogisticRegressionClassifier(
        max_features=max_features,
        ngram_range=ngram_range,
        c_regularization=c_regularization,
        random_state=settings.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate on the holdout
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    metrics = ModelEvaluator.calculate_metrics(y_val, y_pred, y_prob)

    # 6. Save artifacts
    saved_path = model.save(out_dir)

    if val_output_path:
        Path(val_output_path).parent.mkdir(parents=True, exist_ok=True)
        val_df[["title", "text", "label"]].to_csv(val_output_path, index=False)

    if metrics_output_path:
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

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


def _load_params(path: str = "params.yaml") -> dict[str, Any]:
    """Read train settings from params.yaml if present (keeps DVC params meaningful)."""
    p = Path(path)
    if not p.exists():
        return {}
    import yaml

    with open(p, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("train", {})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = _load_params()

    parser = argparse.ArgumentParser(description="Train the baseline fake-news model.")
    parser.add_argument("--data", default="data/raw/news_dataset.csv", help="Training CSV")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-features", type=int, default=params.get("max_features", 5000))
    parser.add_argument("--c", type=float, default=params.get("c_regularization", 1.0))
    parser.add_argument("--ngram-min", type=int, default=params.get("ngram_min", 1))
    parser.add_argument("--ngram-max", type=int, default=params.get("ngram_max", 2))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-out", default="data/processed/val.csv")
    parser.add_argument("--metrics-out", default="artifacts/models/train_metrics.json")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    results = run_training_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        max_features=args.max_features,
        c_regularization=args.c,
        ngram_range=(args.ngram_min, args.ngram_max),
        test_size=args.test_size,
        use_mlflow=args.mlflow,
        val_output_path=args.val_out,
        metrics_output_path=args.metrics_out,
    )
    print("Training Pipeline Execution Finished:")
    print(json.dumps(results, indent=2))
