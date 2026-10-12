"""Training and evaluation pipeline orchestrators."""

from fake_news_detector.pipelines.evaluate import run_evaluation_pipeline
from fake_news_detector.pipelines.train import run_training_pipeline

__all__ = ["run_training_pipeline", "run_evaluation_pipeline"]
