"""Live next-5-trading-days direction prediction for any ticker."""

from __future__ import annotations

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

from src.arimax_model import fit_best_arimax_by_aic
from src.context_stacking import build_stacking_frame, train_context_aware_stacker
from src.data_loader import ProjectConfig, load_price_data
from src.feature_engineering import engineer_features_with_options, get_feature_sets
from src.lstm_model import (
    create_lstm_model,
    scale_frame_with_existing_scaler,
    scale_lstm_frames,
)
from src.stacking import train_logistic_stacker
from src.xgboost_model import create_xgboost_classifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Predict whether a ticker will be higher in 5 trading days.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol such as AAPL, MSFT, NVDA, TSLA, or SPY.")
    parser.add_argument(
        "--start-date",
        default="2013-01-01",
        help="Historical start date for training data.",
    )
    parser.add_argument(
        "--meta-window",
        type=int,
        default=252,
        help="Number of recent labeled rows reserved to train the stackers.",
    )
    return parser.parse_args()


def build_live_config(ticker: str, start_date: str) -> ProjectConfig:
    """Create a config that pulls data up to the current day."""
    tomorrow = (pd.Timestamp.today().normalize() + timedelta(days=1)).strftime("%Y-%m-%d")
    return ProjectConfig(
        ticker=ticker.upper(),
        start_date=start_date,
        end_date=tomorrow,
        test_end=tomorrow,
    )


def prepare_live_frames(
    feature_df: pd.DataFrame,
    meta_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into base-training, stacker-training, and current inference frames."""
    labeled_df = feature_df.dropna(subset=["target", "future_return"]).copy()
    unlabeled_df = feature_df[feature_df["target"].isna()].copy()
    if unlabeled_df.empty:
        raise ValueError("No current unlabeled row found for live inference.")

    meta_window = min(meta_window, max(60, len(labeled_df) // 5))
    if len(labeled_df) <= meta_window + 100:
        raise ValueError("Not enough history to build both base-model and stacker-training windows.")

    base_train_df = labeled_df.iloc[:-meta_window].copy()
    meta_df = labeled_df.iloc[-meta_window:].copy()
    inference_df = unlabeled_df.tail(1).copy()
    return base_train_df, meta_df, inference_df


def train_live_lstm(
    base_train_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train an LSTM and return meta-window and current-row probabilities."""
    scaled_base, _, _, scaler = scale_lstm_frames(
        base_train_df,
        meta_df,
        inference_df,
        feature_columns,
    )
    scaled_meta = scale_frame_with_existing_scaler(meta_df, feature_columns, scaler)
    scaled_inference = scale_frame_with_existing_scaler(inference_df, feature_columns, scaler)

    x_train = []
    y_train = []
    base_values = scaled_base[feature_columns].values
    base_labels = scaled_base["target"].values
    for idx in range(window_size - 1, len(scaled_base)):
        x_train.append(base_values[idx - window_size + 1 : idx + 1])
        y_train.append(base_labels[idx])

    if not x_train:
        raise ValueError("Not enough rows to train the LSTM for live prediction.")

    model = create_lstm_model(window_size, len(feature_columns))
    model.fit(
        np.array(x_train),
        np.array(y_train),
        epochs=20,
        batch_size=32,
        verbose=0,
    )

    meta_combined = pd.concat([scaled_base.tail(window_size - 1), scaled_meta], axis=0)
    meta_sequences = []
    meta_positions = []
    meta_values = meta_combined[feature_columns].values
    for idx in range(window_size - 1, len(meta_combined)):
        meta_sequences.append(meta_values[idx - window_size + 1 : idx + 1])
        meta_positions.append(meta_combined.index[idx])
    meta_probabilities = pd.Series(
        model.predict(np.array(meta_sequences), verbose=0).ravel(),
        index=meta_positions,
    ).loc[meta_df.index]

    inference_combined = pd.concat([scaled_base.tail(window_size - 1), scaled_meta, scaled_inference], axis=0)
    inference_values = inference_combined[feature_columns].values
    inference_sequence = inference_values[-window_size:]
    inference_probability = model.predict(np.array([inference_sequence]), verbose=0).ravel()
    return meta_probabilities.values, inference_probability


def main() -> None:
    args = parse_args()
    config = build_live_config(args.ticker, args.start_date)
    feature_sets = get_feature_sets()

    raw_df = load_price_data(config)
    feature_df = engineer_features_with_options(
        raw_df,
        config,
        drop_target_na=False,
        restrict_to_test_end=False,
    )

    base_train_df, meta_df, inference_df = prepare_live_frames(feature_df, args.meta_window)

    xgb_model = create_xgboost_classifier()
    xgb_model.fit(base_train_df[feature_sets["xgboost"]], base_train_df["target"])
    meta_xgb_prob = xgb_model.predict_proba(meta_df[feature_sets["xgboost"]])[:, 1]
    inference_xgb_prob = xgb_model.predict_proba(inference_df[feature_sets["xgboost"]])[:, 1]

    arimax_model, arimax_order, _ = fit_best_arimax_by_aic(
        base_train_df["future_return"],
        base_train_df[feature_sets["arimax_exog"]],
    )
    meta_arimax_raw = arimax_model.get_forecast(
        steps=len(meta_df),
        exog=meta_df[feature_sets["arimax_exog"]],
    ).predicted_mean.values
    inference_arimax_raw = arimax_model.get_forecast(
        steps=1,
        exog=inference_df[feature_sets["arimax_exog"]],
    ).predicted_mean.values
    meta_arimax_prob = 1 / (1 + np.exp(-meta_arimax_raw))
    inference_arimax_prob = 1 / (1 + np.exp(-inference_arimax_raw))

    meta_lstm_prob, inference_lstm_prob = train_live_lstm(
        base_train_df,
        meta_df,
        inference_df,
        feature_sets["lstm"],
        config.lstm_window,
    )

    meta_base = pd.DataFrame(
        {
            "xgb_prob": meta_xgb_prob,
            "lstm_prob": meta_lstm_prob,
            "arimax_pred": meta_arimax_prob,
        },
        index=meta_df.index,
    )
    inference_base = pd.DataFrame(
        {
            "xgb_prob": inference_xgb_prob,
            "lstm_prob": inference_lstm_prob,
            "arimax_pred": inference_arimax_prob,
        },
        index=inference_df.index,
    )

    logistic_results = train_logistic_stacker(
        meta_base,
        meta_df["target"],
        inference_base,
        pd.Series([0], index=inference_df.index),
    )

    meta_context = meta_df[feature_sets["context"]]
    inference_context = inference_df[feature_sets["context"]]
    context_results = train_context_aware_stacker(
        build_stacking_frame(meta_base, meta_context),
        meta_df["target"],
        build_stacking_frame(inference_base, inference_context),
        pd.Series([0], index=inference_df.index),
    )

    latest_date = inference_df.index[-1].strftime("%Y-%m-%d")
    latest_close = float(inference_df["Close"].iloc[-1])
    context_probability = float(context_results["test_probabilities"][0])
    logistic_probability = float(logistic_results["test_probabilities"][0])

    print(f"Ticker: {config.ticker}")
    print(f"Latest market date used: {latest_date}")
    print(f"Latest adjusted close: {latest_close:.2f}")
    print(f"Forecast horizon: {config.forecast_horizon} trading days")
    print(f"Selected ARIMAX order: {arimax_order}")
    print("")
    print("Base model probabilities (higher after 5 trading days):")
    print(f"  XGBoost: {float(inference_xgb_prob[0]):.4f}")
    print(f"  LSTM: {float(inference_lstm_prob[0]):.4f}")
    print(f"  ARIMAX: {float(inference_arimax_prob[0]):.4f}")
    print("")
    print("Ensemble probabilities:")
    print(f"  Logistic stacking: {logistic_probability:.4f}")
    print(f"  Context-aware XGBoost stacking: {context_probability:.4f}")
    print("")
    print(
        "Final call: "
        + ("UP in 5 trading days" if context_probability >= 0.5 else "NOT higher in 5 trading days")
    )


if __name__ == "__main__":
    main()
