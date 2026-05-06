"""Data access and split configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


RANDOM_SEED = 42


@dataclass(frozen=True)
class ProjectConfig:
    """Shared project configuration for reproducible experiments."""

    ticker: str = "SPY"
    start_date: str = "2013-01-01"
    end_date: str = "2024-01-01"
    train_end: str = "2021-12-31"
    validation_end: str = "2022-12-31"
    test_end: str = "2024-01-01"
    forecast_horizon: int = 5
    lstm_window: int = 20
    results_dir: Path = Path("results")
    figures_dir: Path = Path("results/figures")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns to a single level."""
    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for column in df.columns.to_flat_index():
            values = [str(item) for item in column if item not in ("", None)]
            flattened.append("_".join(values))
        df.columns = flattened
    return df


def _select_single_ticker_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance outputs so columns match a single-ticker schema."""
    rename_map: dict[str, str] = {}
    for column in df.columns:
        if column.endswith(f"_{ticker}"):
            rename_map[column] = column.replace(f"_{ticker}", "")
        elif column.startswith(f"{ticker}_"):
            rename_map[column] = column.replace(f"{ticker}_", "")
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_price_data(config: ProjectConfig) -> pd.DataFrame:
    """Download and normalize OHLCV data from yfinance."""
    data = yf.download(
        tickers=config.ticker,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No data returned for ticker {config.ticker}.")

    data = _flatten_columns(data)
    data = _select_single_ticker_columns(data, config.ticker)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    if "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]
    elif "Close" not in data.columns:
        raise ValueError("Expected either 'Adj Close' or 'Close' in downloaded data.")

    expected = ["Open", "High", "Low", "Close", "Volume"]
    available = [column for column in expected if column in data.columns]
    cleaned = data[available].copy()
    cleaned = cleaned.ffill().dropna()
    return cleaned


def add_split_labels(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Attach chronological split labels used throughout the project."""
    split_df = df.copy()
    split_df["split"] = "test"
    split_df.loc[split_df.index <= pd.Timestamp(config.train_end), "split"] = "train"
    split_df.loc[
        (split_df.index > pd.Timestamp(config.train_end))
        & (split_df.index <= pd.Timestamp(config.validation_end)),
        "split",
    ] = "validation"
    split_df.loc[split_df.index >= pd.Timestamp(config.test_end), "split"] = "post_test"
    return split_df[split_df["split"] != "post_test"].copy()


def ensure_project_dirs(config: ProjectConfig) -> None:
    """Create results directories when needed."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

