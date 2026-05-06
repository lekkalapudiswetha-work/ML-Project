"""Metrics, plots, and simple financial backtesting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, object]:
    """Return a consistent classification summary."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
    return metrics


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap using matplotlib only."""
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_strategy_frame(
    df: pd.DataFrame,
    prediction_column: str,
    future_return_column: str = "future_return",
) -> pd.DataFrame:
    """Compute long/flat strategy returns and buy-and-hold benchmark."""
    strategy_df = df.copy()
    strategy_df["signal"] = strategy_df[prediction_column].astype(int)
    strategy_df["strategy_return"] = strategy_df["signal"] * strategy_df[future_return_column]
    strategy_df["buy_hold_return"] = strategy_df[future_return_column]
    strategy_df["strategy_cumulative"] = (1 + strategy_df["strategy_return"]).cumprod() - 1
    strategy_df["buy_hold_cumulative"] = (1 + strategy_df["buy_hold_return"]).cumprod() - 1
    return strategy_df


def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    """Compute an annualized Sharpe ratio."""
    returns = returns.dropna()
    if returns.empty or np.isclose(returns.std(), 0):
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(annualization)


def save_equity_curve_plot(
    strategy_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Plot strategy and buy-and-hold cumulative returns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy_df.index, strategy_df["strategy_cumulative"], label="Strategy")
    ax.plot(strategy_df.index, strategy_df["buy_hold_cumulative"], label="Buy and Hold")
    ax.set_title(title)
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_prediction_comparison_plot(
    df: pd.DataFrame,
    probability_columns: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Compare probability signals across models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for column in probability_columns:
        ax.plot(df.index, df[column], label=column)
    ax.set_title(title)
    ax.set_ylabel("Predicted Probability")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_model_comparison_plot(
    metrics_df: pd.DataFrame,
    split: str,
    output_path: Path,
) -> None:
    """Save a bar chart comparing model accuracy and F1."""
    subset = metrics_df[metrics_df["split"] == split]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(subset))
    width = 0.35
    ax.bar(x - width / 2, subset["accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, subset["f1"], width, label="F1")
    ax.set_xticks(x, subset["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"Model Comparison ({split.title()})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def append_metric_row(
    rows: list[dict[str, object]],
    split: str,
    model_name: str,
    metrics: dict[str, object],
    sharpe: float,
    cumulative_return: float,
) -> None:
    """Append a standardized metric row for result tables."""
    rows.append(
        {
            "split": split,
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "sharpe_ratio": sharpe,
            "cumulative_return": cumulative_return,
        }
    )

