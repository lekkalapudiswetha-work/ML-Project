"""Generate only the synthetic price-path figure used in the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.synthetic_data import (
    SyntheticConfig,
    generate_linear_regime,
    generate_nonlinear_threshold_regime,
    generate_regime_switching_regime,
    generate_sequential_regime,
)


def main() -> None:
    config = SyntheticConfig(train_size=1200, validation_size=300, test_size=300)
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "Linear": generate_linear_regime(noise_std=0.015, config=config),
        "Nonlinear": generate_nonlinear_threshold_regime(noise_std=0.015, config=config),
        "Sequential": generate_sequential_regime(noise_std=0.015, config=config),
        "Regime Switching": generate_regime_switching_regime(
            noise_std_low=0.0075,
            noise_std_high=0.015,
            config=config,
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, (title, frame) in zip(axes.ravel(), frames.items()):
        ax.plot(frame.index, frame["Close"], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Synthetic Price")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path = output_dir / "synthetic_price_paths.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
