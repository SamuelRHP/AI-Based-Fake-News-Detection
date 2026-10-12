"""Classification evaluation metrics for fake news detection models."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Calculates comprehensive classification performance metrics."""

    @staticmethod
    def calculate_metrics(
        y_true: list[int] | np.ndarray,
        y_pred: list[int] | np.ndarray,
        y_prob: list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute standard evaluation metrics.

        Args:
            y_true: Ground truth binary labels (0=Real, 1=Fake).
            y_pred: Predicted binary labels (0=Real, 1=Fake).
            y_prob: Predicted probability of positive class (Fake).

        Returns:
            Dictionary containing accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
        """
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        roc_auc: float | None = None
        if y_prob is not None:
            try:
                # Handle 2D or 1D probability input
                prob_arr = np.asarray(y_prob)
                probs = prob_arr[:, 1] if prob_arr.ndim == 2 else prob_arr
                roc_auc = float(roc_auc_score(y_true, probs))
            except Exception:
                roc_auc = None

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": cm,
            "sample_count": len(y_true),
        }
