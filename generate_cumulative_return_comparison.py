"""Generate a report-ready cumulative return comparison figure for SPY."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.arimax_model import train_arimax_model
from src.context_stacking import build_stacking_frame, train_context_aware_stacker
from src.data_loader import ProjectConfig, ensure_project_dirs, load_price_data
from src.evaluation import build_strategy_frame
from src.feature_engineering import engineer_features, get_feature_sets, split_frame
from src.lstm_model import train_lstm_model
from src.stacking import train_logistic_stacker
from src.xgboost_model import train_xgboost_model


def _align_lstm_split(
    frame: pd.DataFrame,
    probabilities,
    predictions,
    index: pd.Index,
) -> pd.DataFrame:
    aligned = frame.loc[index].copy()
    aligned["probability"] = probabilities
    aligned["prediction"] = predictions
    return aligned


def main() -> None:
    config = ProjectConfig()
    ensure_project_dirs(config)

    raw_df = load_price_data(config)
    feature_df = engineer_features(raw_df, config)
    feature_sets = get_feature_sets()
    train_df, validation_df, test_df = split_frame(feature_df)

    xgb_results = train_xgboost_model(train_df, validation_df, test_df, feature_sets["xgboost"])
    arimax_results = train_arimax_model(train_df, validation_df, test_df, feature_sets["arimax_exog"])
    lstm_results = train_lstm_model(train_df, validation_df, test_df, feature_sets["lstm"], config.lstm_window)

    lstm_validation_df = _align_lstm_split(
        validation_df,
        lstm_results["validation_probabilities"],
        lstm_results["validation_predictions"],
        lstm_results["validation_index"],
    )
    lstm_test_df = _align_lstm_split(
        test_df,
        lstm_results["test_probabilities"],
        lstm_results["test_predictions"],
        lstm_results["test_index"],
    )

    validation_base = pd.DataFrame(
        {
            "xgb_prob": pd.Series(xgb_results["validation_probabilities"], index=validation_df.index),
            "lstm_prob": lstm_validation_df["probability"],
            "arimax_pred": pd.Series(arimax_results["validation_probabilities"], index=validation_df.index),
        }
    ).dropna()
    test_base = pd.DataFrame(
        {
            "xgb_prob": pd.Series(xgb_results["test_probabilities"], index=test_df.index),
            "lstm_prob": lstm_test_df["probability"],
            "arimax_pred": pd.Series(arimax_results["test_probabilities"], index=test_df.index),
        }
    ).dropna()

    validation_context = validation_df.loc[validation_base.index, feature_sets["context"]]
    test_context = test_df.loc[test_base.index, feature_sets["context"]]
    validation_target = validation_df.loc[validation_base.index, "target"]
    test_target = test_df.loc[test_base.index, "target"]

    logistic_results = train_logistic_stacker(validation_base, validation_target, test_base, test_target)
    context_results = train_context_aware_stacker(
        build_stacking_frame(validation_base, validation_context),
        validation_target,
        build_stacking_frame(test_base, test_context),
        test_target,
    )

    comparison_df = test_df.copy()
    comparison_df["xgboost_pred"] = xgb_results["test_predictions"]
    comparison_df["arimax_pred"] = arimax_results["test_predictions"]
    comparison_df.loc[lstm_test_df.index, "lstm_pred"] = lstm_test_df["prediction"]
    comparison_df.loc[lstm_test_df.index, "logistic_stacking_pred"] = logistic_results["test_predictions"]
    comparison_df.loc[lstm_test_df.index, "context_stacking_pred"] = context_results["test_predictions"]

    strategy_columns = {
        "XGBoost": "xgboost_pred",
        "ARIMAX": "arimax_pred",
        "LSTM": "lstm_pred",
        "Logistic Stacking": "logistic_stacking_pred",
        "Context-Aware Stacking": "context_stacking_pred",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, prediction_column in strategy_columns.items():
        strategy_df = build_strategy_frame(comparison_df.dropna(subset=[prediction_column]), prediction_column)
        ax.plot(strategy_df.index, strategy_df["strategy_cumulative"], label=label)

    buy_hold_df = build_strategy_frame(comparison_df.dropna(subset=["xgboost_pred"]), "xgboost_pred")
    ax.plot(buy_hold_df.index, buy_hold_df["buy_hold_cumulative"], linestyle="--", color="black", label="Buy and Hold")

    ax.set_title("SPY Test Period Cumulative Return Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path("results/figures/spy_cumulative_return_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
