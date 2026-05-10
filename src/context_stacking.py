"""Context-aware stacking using base model outputs and regime features."""

from __future__ import annotations

import math

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


def _time_series_meta_split(
    features: pd.DataFrame,
    target: pd.Series,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a chronological split for stacker regularization and early stopping."""
    split_idx = max(20, math.floor(len(features) * train_fraction))
    split_idx = min(split_idx, len(features) - 1)
    train_x = features.iloc[:split_idx].copy()
    holdout_x = features.iloc[split_idx:].copy()
    train_y = target.iloc[:split_idx].copy()
    holdout_y = target.iloc[split_idx:].copy()
    return train_x, holdout_x, train_y, holdout_y


def train_context_aware_stacker(
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> dict[str, object]:
    """Train the primary XGBoost-based adaptive stacker."""
    stack_train_x, stack_holdout_x, stack_train_y, stack_holdout_y = _time_series_meta_split(
        validation_features,
        validation_target,
    )

    model = XGBClassifier(
        n_estimators=75,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=4,
        gamma=0.8,
        reg_alpha=1.0,
        reg_lambda=8.0,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        early_stopping_rounds=15,
    )
    model.fit(
        stack_train_x,
        stack_train_y,
        eval_set=[(stack_holdout_x, stack_holdout_y)],
        verbose=False,
    )

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
