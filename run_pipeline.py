"""End-to-end runner for the stock forecasting project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.arimax_model import train_arimax_model
from src.context_stacking import build_stacking_frame, train_context_aware_stacker
from src.data_loader import ProjectConfig, ensure_project_dirs, load_price_data
from src.evaluation import (
    append_metric_row,
    build_strategy_frame,
    save_confusion_matrix_plot,
    save_equity_curve_plot,
    save_model_comparison_plot,
    save_prediction_comparison_plot,
    sharpe_ratio,
)
from src.feature_engineering import engineer_features, get_feature_sets, split_frame
from src.lstm_model import save_training_curve_plot, train_lstm_model
from src.stacking import train_logistic_stacker
from src.xgboost_model import save_feature_importance_plot, train_xgboost_model


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

    xgb_results = train_xgboost_model(
        train_df,
        validation_df,
        test_df,
        feature_sets["xgboost"],
    )
    arimax_results = train_arimax_model(
        train_df,
        validation_df,
        test_df,
        feature_sets["arimax_exog"],
    )
    lstm_results = train_lstm_model(
        train_df,
        validation_df,
        test_df,
        feature_sets["lstm"],
        config.lstm_window,
    )

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

    logistic_stacking_features = validation_base.copy()
    logistic_test_features = test_base.copy()

    context_validation_features = build_stacking_frame(validation_base, validation_context)
    context_test_features = build_stacking_frame(test_base, test_context)

    logistic_results = train_logistic_stacker(
        logistic_stacking_features,
        validation_target,
        logistic_test_features,
        test_target,
    )
    context_results = train_context_aware_stacker(
        context_validation_features,
        validation_target,
        context_test_features,
        test_target,
    )

    save_feature_importance_plot(
        xgb_results["model"],
        feature_sets["xgboost"],
        config.figures_dir / "xgboost_feature_importance.png",
    )
    save_training_curve_plot(
        lstm_results["history"],
        config.figures_dir / "lstm_training_curves.png",
    )

    prediction_frames = {
        "validation": validation_df.copy(),
        "test": test_df.copy(),
    }
    prediction_frames["validation"]["xgboost_pred"] = xgb_results["validation_predictions"]
    prediction_frames["validation"]["arimax_pred"] = arimax_results["validation_predictions"]
    prediction_frames["test"]["xgboost_pred"] = xgb_results["test_predictions"]
    prediction_frames["test"]["arimax_pred"] = arimax_results["test_predictions"]

    prediction_frames["validation"].loc[lstm_validation_df.index, "lstm_pred"] = lstm_validation_df["prediction"]
    prediction_frames["validation"].loc[lstm_validation_df.index, "logistic_stacking_pred"] = logistic_results[
        "validation_predictions"
    ]
    prediction_frames["validation"].loc[lstm_validation_df.index, "context_xgboost_pred"] = context_results[
        "validation_predictions"
    ]

    prediction_frames["test"].loc[lstm_test_df.index, "lstm_pred"] = lstm_test_df["prediction"]
    prediction_frames["test"].loc[lstm_test_df.index, "logistic_stacking_pred"] = logistic_results[
        "test_predictions"
    ]
    prediction_frames["test"].loc[lstm_test_df.index, "context_xgboost_pred"] = context_results[
        "test_predictions"
    ]

    save_prediction_comparison_plot(
        context_validation_features.assign(
            logistic_prob=logistic_results["validation_probabilities"],
            context_prob=context_results["validation_probabilities"],
        ),
        ["xgb_prob", "lstm_prob", "arimax_pred", "logistic_prob", "context_prob"],
        config.figures_dir / "validation_prediction_comparison.png",
        "Validation Prediction Comparison",
    )

    metrics_rows: list[dict[str, object]] = []
    model_metric_pairs = {
        "XGBoost": xgb_results,
        "ARIMAX": arimax_results,
        "LSTM": {
            "validation_metrics": lstm_results["validation_metrics"],
            "test_metrics": lstm_results["test_metrics"],
        },
        "Logistic Stacking": logistic_results,
        "Context-Aware XGBoost Stacking": context_results,
    }

    prediction_columns = {
        "XGBoost": "xgboost_pred",
        "ARIMAX": "arimax_pred",
        "LSTM": "lstm_pred",
        "Logistic Stacking": "logistic_stacking_pred",
        "Context-Aware XGBoost Stacking": "context_xgboost_pred",
    }

    for split_name, frame in prediction_frames.items():
        for model_name, prediction_column in prediction_columns.items():
            strategy_frame = build_strategy_frame(frame.dropna(subset=[prediction_column]), prediction_column)
            save_equity_curve_plot(
                strategy_frame,
                f"{model_name} {split_name.title()} Equity Curve",
                config.figures_dir
                / f"{split_name}_{model_name.lower().replace(' ', '_').replace('-', '_')}_equity_curve.png",
            )
            metrics = model_metric_pairs[model_name][f"{split_name}_metrics"]
            append_metric_row(
                metrics_rows,
                split_name,
                model_name,
                metrics,
                sharpe_ratio(strategy_frame["strategy_return"]),
                strategy_frame["strategy_cumulative"].iloc[-1],
            )
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                f"{model_name} {split_name.title()} Confusion Matrix",
                config.figures_dir
                / f"{split_name}_{model_name.lower().replace(' ', '_').replace('-', '_')}_confusion_matrix.png",
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(config.results_dir / "metrics.csv", index=False)
    save_model_comparison_plot(
        metrics_df,
        "validation",
        config.figures_dir / "validation_model_comparison.png",
    )
    save_model_comparison_plot(
        metrics_df,
        "test",
        config.figures_dir / "test_model_comparison.png",
    )

    validation_base.to_csv(config.results_dir / "validation_base_predictions.csv", index=True)
    test_base.to_csv(config.results_dir / "test_base_predictions.csv", index=True)
    arimax_results["candidate_summary"].to_csv(config.results_dir / "arimax_model_selection.csv", index=False)

    print("Pipeline complete. Metrics saved to results/metrics.csv")


if __name__ == "__main__":
    main()
