"""Feature engineering for time-series stock forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import ProjectConfig, add_split_labels


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series) -> pd.Series:
    """Compute MACD line."""
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26


def engineer_features(raw_df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Create modeling features and both classification/regression targets."""
    return engineer_features_with_options(raw_df, config)


def engineer_features_with_options(
    raw_df: pd.DataFrame,
    config: ProjectConfig,
    drop_target_na: bool = True,
    restrict_to_test_end: bool = True,
) -> pd.DataFrame:
    """Create modeling features with configurable retention of unlabeled rows."""
    df = raw_df.copy()

    df["returns"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["trend_signal"] = df["MA5"] - df["MA20"]
    df["ma_cross"] = (df["MA5"] > df["MA20"]).astype(int)

    df["vol5"] = df["returns"].rolling(5).std()
    df["vol10"] = df["returns"].rolling(10).std()
    df["vol20"] = df["returns"].rolling(20).std()

    df["rsi"] = compute_rsi(df["Close"])
    df["macd"] = compute_macd(df["Close"])
    df["momentum5"] = (df["Close"] / df["Close"].shift(5)) - 1

    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["returns"].shift(lag)

    df["volume_ma20"] = df["Volume"].rolling(20).mean()
    df["volume_change"] = df["Volume"].pct_change()

    horizon = config.forecast_horizon
    future_close = df["Close"].shift(-horizon)
    df["future_return"] = (future_close / df["Close"]) - 1
    df["target"] = np.where(
        future_close.isna(),
        np.nan,
        (future_close > df["Close"]).astype(int),
    )

    df = add_split_labels(df, config, restrict_to_test_end=restrict_to_test_end)
    df = df.replace([np.inf, -np.inf], np.nan)
    required_features = sorted({feature for features in get_feature_sets().values() for feature in features})
    df = df.dropna(subset=required_features).copy()
    if drop_target_na:
        df = df.dropna(subset=["future_return", "target"]).copy()
        df["target"] = df["target"].astype(int)
    return df


def get_feature_sets() -> dict[str, list[str]]:
    """Feature groups used by the different models and stackers."""
    core_features = [
        "MA5",
        "MA20",
        "MA50",
        "trend_signal",
        "ma_cross",
        "vol5",
        "vol10",
        "vol20",
        "rsi",
        "macd",
        "momentum5",
        "returns",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_5",
        "volume_ma20",
        "volume_change",
    ]
    lstm_features = [
        "trend_signal",
        "vol10",
        "rsi",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_5",
    ]
    arimax_exog = ["trend_signal", "vol10", "rsi"]
    context_features = [
        "vol10",
        "vol20",
        "rsi",
        "trend_signal",
        "momentum5",
        "macd",
        "volume_change",
    ]
    return {
        "xgboost": core_features,
        "lstm": lstm_features,
        "arimax_exog": arimax_exog,
        "context": context_features,
    }


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test DataFrames."""
    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    return train, validation, test
