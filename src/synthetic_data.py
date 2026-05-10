"""Synthetic data generators for controlled forecasting experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import RANDOM_SEED


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    train_size: int = 2000
    validation_size: int = 500
    test_size: int = 500
    start_date: str = "2013-01-01"
    initial_price: float = 100.0
    base_volume: float = 1_000_000.0

    @property
    def total_size(self) -> int:
        return self.train_size + self.validation_size + self.test_size


def _rng(seed_offset: int = 0) -> np.random.Generator:
    """Create a deterministic random generator."""
    return np.random.default_rng(RANDOM_SEED + seed_offset)


def _returns_to_ohlcv_frame(
    returns: np.ndarray,
    volumes: np.ndarray,
    config: SyntheticConfig,
    regime_state: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert simulated returns and volume into an OHLCV-style DataFrame."""
    close = [config.initial_price]
    for current_return in returns:
        close.append(close[-1] * (1 + current_return))
    close = np.array(close[1:])

    open_prices = np.concatenate([[config.initial_price], close[:-1]])
    intraday_scale = np.maximum(np.abs(returns), 0.002)
    high = np.maximum(open_prices, close) * (1 + 0.5 * intraday_scale)
    low = np.minimum(open_prices, close) * (1 - 0.5 * intraday_scale)

    dates = pd.bdate_range(config.start_date, periods=len(returns))
    frame = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.maximum(volumes, 1.0),
        },
        index=dates,
    )
    if regime_state is not None:
        frame["regime_state"] = regime_state
    return frame


def generate_linear_regime(
    noise_std: float = 0.01,
    config: SyntheticConfig | None = None,
    seed_offset: int = 0,
) -> pd.DataFrame:
    """Generate a linear autoregressive regime with exogenous signal."""
    config = config or SyntheticConfig()
    rng = _rng(seed_offset)
    total_size = config.total_size

    exog_signal = rng.normal(0, 1, total_size)
    noise = rng.normal(0, noise_std, total_size)
    returns = np.zeros(total_size)

    for idx in range(1, total_size):
        returns[idx] = 0.25 * returns[idx - 1] + 0.015 * exog_signal[idx] + noise[idx]

    volume_noise = rng.normal(0, 0.08, total_size)
    volumes = config.base_volume * (1 + 0.35 * np.abs(returns) + volume_noise)
    return _returns_to_ohlcv_frame(returns, volumes, config)


def generate_nonlinear_threshold_regime(
    noise_std: float = 0.01,
    config: SyntheticConfig | None = None,
    seed_offset: int = 100,
) -> pd.DataFrame:
    """Generate a nonlinear threshold-driven regime."""
    config = config or SyntheticConfig()
    rng = _rng(seed_offset)
    total_size = config.total_size

    latent_signal = rng.normal(0, 1, total_size)
    noise = rng.normal(0, noise_std, total_size)
    returns = np.zeros(total_size)

    for idx in range(1, total_size):
        threshold_effect = 0.02 if latent_signal[idx] > 0 else -0.018
        feedback = 0.2 * returns[idx - 1] if latent_signal[idx] > 0 else -0.08 * returns[idx - 1]
        returns[idx] = feedback + threshold_effect + noise[idx]

    volume_noise = rng.normal(0, 0.12, total_size)
    volumes = config.base_volume * (1 + 0.5 * np.abs(returns) + volume_noise)
    return _returns_to_ohlcv_frame(returns, volumes, config)


def generate_sequential_regime(
    noise_std: float = 0.01,
    config: SyntheticConfig | None = None,
    seed_offset: int = 200,
) -> pd.DataFrame:
    """Generate a persistent latent-state regime with memory."""
    config = config or SyntheticConfig()
    rng = _rng(seed_offset)
    total_size = config.total_size

    regime_state = np.zeros(total_size, dtype=int)
    for idx in range(1, total_size):
        if rng.uniform() < 0.9:
            regime_state[idx] = regime_state[idx - 1]
        else:
            regime_state[idx] = 1 - regime_state[idx - 1]

    noise = rng.normal(0, noise_std, total_size)
    returns = np.zeros(total_size)
    mu = np.where(regime_state == 1, 0.012, -0.008)
    for idx in range(2, total_size):
        returns[idx] = mu[idx] + 0.35 * returns[idx - 1] + 0.10 * returns[idx - 2] + noise[idx]

    volume_noise = rng.normal(0, 0.1, total_size)
    volumes = config.base_volume * (1 + 0.4 * np.abs(returns) + 0.15 * regime_state + volume_noise)
    return _returns_to_ohlcv_frame(returns, volumes, config, regime_state=regime_state)


def generate_regime_switching_regime(
    noise_std_low: float = 0.008,
    noise_std_high: float = 0.025,
    config: SyntheticConfig | None = None,
    seed_offset: int = 300,
) -> pd.DataFrame:
    """Generate a regime-switching series combining linear and nonlinear behavior."""
    config = config or SyntheticConfig()
    rng = _rng(seed_offset)
    total_size = config.total_size

    regime_state = np.zeros(total_size, dtype=int)
    for idx in range(1, total_size):
        if rng.uniform() < 0.92:
            regime_state[idx] = regime_state[idx - 1]
        else:
            regime_state[idx] = 1 - regime_state[idx - 1]

    latent_signal = rng.normal(0, 1, total_size)
    returns = np.zeros(total_size)

    for idx in range(1, total_size):
        if regime_state[idx] == 0:
            noise = rng.normal(0, noise_std_low)
            returns[idx] = 0.30 * returns[idx - 1] + 0.012 * latent_signal[idx] + noise
        else:
            noise = rng.normal(0, noise_std_high)
            threshold_effect = 0.025 if latent_signal[idx] > 0 else -0.02
            returns[idx] = 0.08 * returns[idx - 1] + threshold_effect + noise

    volume_noise = rng.normal(0, 0.14, total_size)
    volumes = config.base_volume * (
        1 + 0.35 * np.abs(returns) + 0.25 * regime_state + volume_noise
    )
    return _returns_to_ohlcv_frame(returns, volumes, config, regime_state=regime_state)

