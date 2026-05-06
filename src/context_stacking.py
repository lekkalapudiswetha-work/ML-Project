"""Context-aware stacking using base model outputs and regime features."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier

from src.data_loader import RANDOM_SEED
from src.evaluation import compute_classification_metrics


def build_stacking_frame(
    base_predictions: pd.DataFrame,
    context_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge base probabilities with market context features."""
    return pd.concat([base_predictions, context_features], axis=1)


def train_context_aware_stacker(
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> dict[str, object]:
    """Train the primary XGBoost-based adaptive stacker."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
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

