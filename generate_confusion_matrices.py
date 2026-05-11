"""Generate report-ready confusion matrix plots for the stock forecasting project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.arimax_model import train_arimax_model
from src.context_stacking import build_stacking_frame, train_context_aware_stacker
from src.data_loader import ProjectConfig, ensure_project_dirs, load_price_data
from src.evaluation import save_confusion_matrix_plot
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
    parser = argparse.ArgumentParser(description="Generate report-ready confusion matrix plots.")
    parser.add_argument(
        "--model",
        default="LSTM",
        choices=[
            "XGBoost",
            "ARIMAX",
            "LSTM",
            "Logistic Stacking",
            "Context-Aware XGBoost Stacking",
            "all",
        ],
        help="Which model confusion matrix to save. Defaults to LSTM.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["validation", "test", "both"],
        help="Which split to save. Defaults to test.",
    )
    args = parser.parse_args()

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

    logistic_results = train_logistic_stacker(
        validation_base,
        validation_target,
        test_base,
        test_target,
    )
    context_results = train_context_aware_stacker(
        build_stacking_frame(validation_base, validation_context),
        validation_target,
        build_stacking_frame(test_base, test_context),
        test_target,
    )

    model_results = {
        "XGBoost": xgb_results,
        "ARIMAX": arimax_results,
        "LSTM": lstm_results,
        "Logistic Stacking": logistic_results,
        "Context-Aware XGBoost Stacking": context_results,
    }

    target_models = list(model_results) if args.model == "all" else [args.model]
    target_splits = ["validation", "test"] if args.split == "both" else [args.split]

    for model_name in target_models:
        results = model_results[model_name]
        for split_name in target_splits:
            metrics = results[f"{split_name}_metrics"]
            safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
            output_path = config.figures_dir / f"{split_name}_{safe_name}_confusion_matrix.png"
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                f"{model_name} {split_name.title()} Confusion Matrix",
                output_path,
            )
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
