"""Run synthetic experiments for controlled model-behavior analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.arimax_model import train_arimax_model
from src.context_stacking import build_stacking_frame, train_context_aware_stacker
from src.data_loader import ProjectConfig, ensure_project_dirs
from src.evaluation import append_metric_row, save_model_comparison_plot
from src.feature_engineering import engineer_features_with_options, get_feature_sets, split_frame
from src.lstm_model import train_lstm_model
from src.stacking import train_logistic_stacker
from src.synthetic_data import (
    SyntheticConfig,
    generate_linear_regime,
    generate_nonlinear_threshold_regime,
    generate_regime_switching_regime,
    generate_sequential_regime,
)
from src.xgboost_model import train_xgboost_model


NOISE_LEVELS = {
    "low": 0.005,
    "medium": 0.015,
    "high": 0.03,
}

SAMPLE_SIZES = {
    "small": SyntheticConfig(train_size=700, validation_size=200, test_size=200),
    "medium": SyntheticConfig(train_size=1200, validation_size=300, test_size=300),
    "large": SyntheticConfig(train_size=2000, validation_size=500, test_size=500),
}


def _synthetic_project_config(frame: pd.DataFrame) -> ProjectConfig:
    """Construct a split config tied to the synthetic frame length."""
    dates = frame.index
    train_end = dates[SAMPLE_SIZES["large"].train_size - 1] if len(dates) >= SAMPLE_SIZES["large"].total_size else None
    if train_end is None:
        total = len(dates)
        train_end = dates[int(total * 0.7) - 1]
        validation_end = dates[int(total * 0.85) - 1]
        end_date = str(dates[-1] + pd.Timedelta(days=1))
    else:
        validation_end = dates[SAMPLE_SIZES["large"].train_size + SAMPLE_SIZES["large"].validation_size - 1]
        end_date = str(dates[-1] + pd.Timedelta(days=1))

    return ProjectConfig(
        ticker="SYNTH",
        start_date=str(dates[0].date()),
        end_date=end_date,
        train_end=str(pd.Timestamp(train_end).date()),
        validation_end=str(pd.Timestamp(validation_end).date()),
        test_end=end_date,
    )


def _build_config_from_sizes(frame: pd.DataFrame, synthetic_config: SyntheticConfig) -> ProjectConfig:
    """Create a chronological split config from explicit synthetic sizes."""
    dates = frame.index
    train_end = dates[synthetic_config.train_size - 1]
    validation_end = dates[synthetic_config.train_size + synthetic_config.validation_size - 1]
    end_date = str((dates[-1] + pd.Timedelta(days=1)).date())
    return ProjectConfig(
        ticker="SYNTH",
        start_date=str(dates[0].date()),
        end_date=end_date,
        train_end=str(train_end.date()),
        validation_end=str(validation_end.date()),
        test_end=end_date,
    )


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


def _run_single_synthetic_experiment(
    regime_name: str,
    raw_df: pd.DataFrame,
    synthetic_config: SyntheticConfig,
) -> list[dict[str, object]]:
    """Train all models on a single synthetic dataset and return metric rows."""
    config = _build_config_from_sizes(raw_df, synthetic_config)
    feature_df = engineer_features_with_options(raw_df, config, drop_target_na=True, restrict_to_test_end=True)
    train_df, validation_df, test_df = split_frame(feature_df)
    feature_sets = get_feature_sets()

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

    validation_target = validation_df.loc[validation_base.index, "target"]
    test_target = test_df.loc[test_base.index, "target"]
    validation_context = validation_df.loc[validation_base.index, feature_sets["context"]]
    test_context = test_df.loc[test_base.index, feature_sets["context"]]

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

    rows: list[dict[str, object]] = []
    for split_name, metrics_key in [("validation", "validation_metrics"), ("test", "test_metrics")]:
        for model_name, results in [
            ("XGBoost", xgb_results),
            ("ARIMAX", arimax_results),
            ("LSTM", lstm_results),
            ("Logistic Stacking", logistic_results),
            ("Context-Aware XGBoost Stacking", context_results),
        ]:
            metrics = results[metrics_key]
            rows.append(
                {
                    "regime": regime_name,
                    "split": split_name,
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
            )
    return rows


def _plot_example_price_paths(results_dir: Path) -> None:
    """Save one representative price path per synthetic regime."""
    config = SAMPLE_SIZES["medium"]
    example_frames = {
        "Linear": generate_linear_regime(noise_std=NOISE_LEVELS["medium"], config=config),
        "Nonlinear": generate_nonlinear_threshold_regime(noise_std=NOISE_LEVELS["medium"], config=config),
        "Sequential": generate_sequential_regime(noise_std=NOISE_LEVELS["medium"], config=config),
        "Regime Switching": generate_regime_switching_regime(config=config),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, (title, frame) in zip(axes.ravel(), example_frames.items()):
        ax.plot(frame.index, frame["Close"])
        ax.set_title(title)
        ax.set_ylabel("Synthetic Price")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "synthetic_price_paths.png", dpi=200)
    plt.close(fig)


def _plot_noise_sensitivity(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot test accuracy versus noise level for each regime and model."""
    ordered_noise = ["low", "medium", "high"]
    for regime_name in metrics_df["regime"].unique():
        subset = metrics_df[
            (metrics_df["split"] == "test")
            & (metrics_df["regime"] == regime_name)
            & (metrics_df["experiment_type"] == "noise")
        ].copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        for model_name in subset["model"].unique():
            model_subset = subset[subset["model"] == model_name].set_index("condition_label").reindex(ordered_noise)
            ax.plot(ordered_noise, model_subset["accuracy"], marker="o", label=model_name)
        ax.set_title(f"{regime_name}: Test Accuracy vs Noise Level")
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        output_name = regime_name.lower().replace(" ", "_")
        fig.savefig(figures_dir / f"{output_name}_noise_sensitivity.png", dpi=200)
        plt.close(fig)


def _plot_sample_size_sensitivity(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot test accuracy versus sample size."""
    ordered_sizes = ["small", "medium", "large"]
    subset = metrics_df[
        (metrics_df["split"] == "test")
        & (metrics_df["regime"] == "Regime Switching")
        & (metrics_df["experiment_type"] == "sample_size")
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name in subset["model"].unique():
        model_subset = subset[subset["model"] == model_name].set_index("condition_label").reindex(ordered_sizes)
        ax.plot(ordered_sizes, model_subset["accuracy"], marker="o", label=model_name)
    ax.set_title("Regime Switching: Test Accuracy vs Sample Size")
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "regime_switching_sample_size_sensitivity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the full synthetic experiment suite."""
    config = ProjectConfig()
    ensure_project_dirs(config)
    synthetic_results_path = config.results_dir / "synthetic_metrics.csv"

    all_rows: list[dict[str, object]] = []
    regime_generators = {
        "Linear": lambda noise, cfg: generate_linear_regime(noise_std=noise, config=cfg),
        "Nonlinear": lambda noise, cfg: generate_nonlinear_threshold_regime(noise_std=noise, config=cfg),
        "Sequential": lambda noise, cfg: generate_sequential_regime(noise_std=noise, config=cfg),
        "Regime Switching": lambda noise, cfg: generate_regime_switching_regime(
            noise_std_low=max(noise / 2, 0.003),
            noise_std_high=noise,
            config=cfg,
        ),
    }

    for regime_name, generator in regime_generators.items():
        for noise_label, noise_value in NOISE_LEVELS.items():
            synthetic_config = SAMPLE_SIZES["large"]
            raw_df = generator(noise_value, synthetic_config)
            rows = _run_single_synthetic_experiment(regime_name, raw_df, synthetic_config)
            for row in rows:
                row["experiment_type"] = "noise"
                row["condition_label"] = noise_label
                row["noise_value"] = noise_value
                row["sample_size"] = synthetic_config.total_size
            all_rows.extend(rows)

    for size_label, synthetic_config in SAMPLE_SIZES.items():
        raw_df = generate_regime_switching_regime(config=synthetic_config)
        rows = _run_single_synthetic_experiment("Regime Switching", raw_df, synthetic_config)
        for row in rows:
            row["experiment_type"] = "sample_size"
            row["condition_label"] = size_label
            row["noise_value"] = NOISE_LEVELS["medium"]
            row["sample_size"] = synthetic_config.total_size
        all_rows.extend(rows)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(synthetic_results_path, index=False)

    _plot_example_price_paths(config.figures_dir)
    _plot_noise_sensitivity(metrics_df, config.figures_dir)
    _plot_sample_size_sensitivity(metrics_df, config.figures_dir)

    for regime_name in metrics_df["regime"].unique():
        subset = metrics_df[
            (metrics_df["regime"] == regime_name)
            & (metrics_df["split"] == "test")
            & (metrics_df["experiment_type"] == "noise")
            & (metrics_df["condition_label"] == "medium")
        ][["split", "model", "accuracy", "precision", "recall", "f1"]].copy()
        subset.insert(0, "regime", regime_name)
        output_name = regime_name.lower().replace(" ", "_")
        subset.to_csv(config.results_dir / f"{output_name}_synthetic_summary.csv", index=False)

        save_model_comparison_plot(
            subset.assign(split="test"),
            "test",
            config.figures_dir / f"{output_name}_synthetic_model_comparison.png",
        )

    print(f"Synthetic experiments complete. Metrics saved to {synthetic_results_path}")


if __name__ == "__main__":
    main()
