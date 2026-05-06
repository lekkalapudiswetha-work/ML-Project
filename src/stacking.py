"""Static stacking models built from validation predictions."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data_loader import RANDOM_SEED
from src.evaluation import compute_classification_metrics


def train_logistic_stacker(
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> dict[str, object]:
    """Train a simple logistic regression stacker."""
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    model.fit(validation_features, validation_target)

    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    validation_predictions = (validation_probabilities >= 0.5).astype(int)
    test_probabilities = model.predict_proba(test_features)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    return {
        "model": model,
        "validation_probabilities": validation_probabilities,
        "validation_predictions": validation_predictions,
        "validation_metrics": compute_classification_metrics(validation_target, validation_predictions),
        "test_probabilities": test_probabilities,
        "test_predictions": test_predictions,
        "test_metrics": compute_classification_metrics(test_target, test_predictions),
    }

