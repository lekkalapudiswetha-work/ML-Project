"""XGBoost baseline model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

from src.data_loader import RANDOM_SEED
from src.evaluation import compute_classification_metrics


def create_xgboost_classifier() -> XGBClassifier:
    """Construct the shared XGBoost classifier configuration."""
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )


def train_xgboost_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    """Fit XGBoost and return probabilities and metrics for each split."""
    model = create_xgboost_classifier()
    model.fit(train_df[feature_columns], train_df["target"])

    outputs: dict[str, object] = {"model": model}
    for split_name, frame in [("validation", validation_df), ("test", test_df)]:
        probabilities = model.predict_proba(frame[feature_columns])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        outputs[f"{split_name}_probabilities"] = probabilities
        outputs[f"{split_name}_predictions"] = predictions
        outputs[f"{split_name}_metrics"] = compute_classification_metrics(frame["target"], predictions)
    return outputs


def save_feature_importance_plot(
    model: XGBClassifier,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """Save model feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index, importances.values)
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
