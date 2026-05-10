<<<<<<< HEAD
# Stock Market Prediction Using Ensemble Methods

Graduate ML project exploring ensemble methods for financial time series forecasting.

## Models Implemented
- ARIMAX
- XGBoost
- LSTM
- (Upcoming) Stacking Ensemble

## Current Results

| Model | Validation Accuracy |
|------|----------------|
| ARIMAX | 47.8% |
| XGBoost | 53.0% |
| LSTM | 59.6% |

## Features
- Technical indicators
- Lagged returns
- Volatility signals
- Momentum signals

## Goal
Investigate whether combining diverse models improves stock direction prediction.

## Next Steps
- Stacking ensemble
- Financial backtesting
- Risk metrics

Key Findings:

1-day forecasting too noisy
5-day horizon improved signal
LSTM outperformed classical and tree-based models
=======
# Adaptive Ensemble Stock Forecasting with Context-Aware Stacking

Graduate-level machine learning project for forecasting whether a security will be higher after **5 trading days**. The repository is designed as a reproducible research pipeline and portfolio-ready engineering project, combining econometric, tree-based, and deep learning models with a regime-aware meta-learner. It now supports:

- real-data experiments on **SPY** and configurable tickers
- synthetic experiments for controlled model-behavior analysis
- live next-5-trading-days directional prediction for any chosen ticker

## Project Overview

This project predicts the binary target:

```python
target = (Close.shift(-5) > Close).astype(int)
```

It also models the 5-day forward return:

```python
future_return = (Close.shift(-5) / Close) - 1
```

The initial real dataset is **SPY** from **2013-01-01** to **2024-01-01** using `yfinance` adjusted close data. The architecture is configurable so the same pipeline can also support `AAPL`, `MSFT`, `NVDA`, `TSLA`, `QQQ`, or future multi-ticker extensions.

## Methodology

The repository implements a strict chronological workflow to prevent temporal leakage:

- Train: `2013-01-01` to `2021-12-31`
- Validation: `2022-01-01` to `2022-12-31`
- Test: `2023-01-01` to `2024-01-01`

### Base Models

1. **ARIMAX**
   Models 5-day forward returns using exogenous market indicators:
   - `trend_signal`
   - `vol10`
   - `rsi`

2. **XGBoost**
   Learns nonlinear interactions over engineered trend, volatility, momentum, lag, and volume features.

3. **LSTM**
   Uses 20-day rolling windows over sequential technical indicators:
   - `trend_signal`
   - `vol10`
   - `rsi`
   - `lag_1` to `lag_5`

### Context-Aware Stacking

The primary ensemble is not simple averaging. Instead, it is a **context-aware stacker** that learns how model reliability changes across market regimes.

Meta-model inputs:

- Base model outputs:
  - `xgb_prob`
  - `lstm_prob`
  - `arimax_pred`
- Context features:
  - `vol10`
  - `vol20`
  - `rsi`
  - `trend_signal`
  - `momentum5`
  - `macd`
  - `volume_change`

Two stackers are implemented:

1. `LogisticRegression` stacker
2. `XGBoost` stacker for **primary context-aware adaptive ensembling**

Adaptive stacking is superior to static averaging because it can learn that model skill is conditional:

- In higher volatility, sequence models may capture nonlinear transitions better.
- In smoother, trend-dominated periods, ARIMAX may be more stable.
- In feature-rich nonlinear regimes, XGBoost may dominate.

Instead of manually coding those rules, the stacker learns them directly from validation-period behavior. The current implementation also applies a **regularized, chronology-aware meta-training procedure** to reduce overfitting:

- shallower trees
- stronger L1/L2 penalties
- lower subsampling rates
- chronological holdout within the validation period
- early stopping for the context-aware XGBoost stacker

## Repository Structure

```text
ML-Project/
├── notebooks/
│   ├── 01_data_and_features.ipynb
│   ├── 02_xgboost_baseline.ipynb
│   ├── 03_arimax_baseline.ipynb
│   ├── 04_lstm_baseline.ipynb
│   ├── 05_basic_stacking.ipynb
│   └── 06_context_aware_stacking.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── xgboost_model.py
│   ├── arimax_model.py
│   ├── lstm_model.py
│   ├── stacking.py
│   ├── context_stacking.py
│   ├── evaluation.py
│   └── synthetic_data.py
├── results/
│   ├── figures/
│   └── metrics.csv
├── run_pipeline.py
├── run_synthetic_experiments.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Feature Engineering

Implemented features include:

- Trend:
  - `MA5`, `MA20`, `MA50`
  - `trend_signal = MA5 - MA20`
  - `ma_cross`
- Volatility:
  - `vol5`, `vol10`, `vol20`
- Momentum:
  - `RSI`
  - `MACD`
  - `momentum5`
- Returns:
  - `returns`
- Lagged returns:
  - `lag_1` to `lag_5`
- Volume:
  - `volume_ma20`
  - `volume_change`

NaNs introduced by rolling indicators and forward targets are handled carefully and dropped only after all derived fields are created.

## Evaluation Framework

Each model is evaluated on validation and test sets using:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- Classification report
- Sharpe ratio
- Cumulative returns under a simple long/flat strategy

Trading rule:

```python
if prediction == 1:
    long
else:
    flat
```

Generated artifacts include:

- XGBoost feature importances
- Confusion matrices
- LSTM training curves
- Prediction comparison plots
- Strategy equity curves
- Model comparison charts
- `results/metrics.csv`

## Synthetic Experiments

To satisfy graduate coursework requirements and test model inductive biases under controlled conditions, the project includes a synthetic experiment suite in [src/synthetic_data.py](/Users/swethalekkalapudi/Documents/ML-Project/src/synthetic_data.py) and [run_synthetic_experiments.py](/Users/swethalekkalapudi/Documents/ML-Project/run_synthetic_experiments.py).

Implemented synthetic regimes:

- **Linear autoregressive regime**
  Intended to favor ARIMAX under smooth linear dynamics.
- **Nonlinear threshold regime**
  Intended to favor XGBoost when piecewise nonlinear effects dominate.
- **Sequential latent-state regime**
  Intended to favor LSTM when temporal memory and persistence matter.
- **Regime-switching regime**
  Intended to test whether context-aware stacking can adapt when model reliability changes with market state.

The synthetic suite also evaluates:

- noise sensitivity
- sample-size sensitivity
- cross-model behavior under controlled structure

Outputs include:

- `results/synthetic_metrics.csv`
- synthetic price path plots
- noise sensitivity plots
- sample-size sensitivity plots
- medium-noise synthetic summary tables for each regime

## Model Comparison Table

The pipeline writes the final comparison table to `results/metrics.csv`. The schema is:

| split | model | accuracy | precision | recall | f1 | sharpe_ratio | cumulative_return |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | XGBoost | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |
| validation | ARIMAX | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |
| validation | LSTM | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |
| validation | Logistic Stacking | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |
| validation | Context-Aware XGBoost Stacking | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |
| test | XGBoost | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |
| test | ARIMAX | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |
| test | LSTM | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |
| test | Logistic Stacking | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |
| test | Context-Aware XGBoost Stacking | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

The full real-data pipeline downloads SPY data, engineers features, trains all base learners, fits both stacking models, saves metrics, and writes figures into `results/figures/`.

Note for macOS users: `xgboost` may require `libomp`:

```bash
brew install libomp
```

TensorFlow should be installed with **Python 3.11** rather than Python 3.14.

## Usage

### Real-data training

```bash
python run_pipeline.py
```

### Synthetic experiment suite

```bash
python run_synthetic_experiments.py
```

### Live 5-day direction prediction for any ticker

```bash
python predict.py --ticker AAPL
python predict.py --ticker MSFT
python predict.py --ticker NVDA
python predict.py --ticker SPY
```

Optional:

```bash
python predict.py --ticker AAPL --start-date 2018-01-01
```

This command retrains the base models and stackers on that ticker’s history and outputs the current probability that the asset will be higher in 5 trading days.

## Notebooks

The notebook sequence is organized as a coursework-friendly walkthrough:

1. `01_data_and_features.ipynb`
2. `02_xgboost_baseline.ipynb`
3. `03_arimax_baseline.ipynb`
4. `04_lstm_baseline.ipynb`
5. `05_basic_stacking.ipynb`
6. `06_context_aware_stacking.ipynb`

Each notebook mirrors the production modules in `src/` rather than re-implementing logic inline.

## Results

The project is structured so final metrics and plots are produced deterministically once dependencies are installed and the experiment runners are executed. This keeps the repository lightweight while preserving reproducibility.

### Real-Data Observations

In the current SPY experiment:

- the **LSTM** is the strongest held-out test model
- **XGBoost** is a competitive nonlinear baseline
- **ARIMAX** provides a useful interpretable econometric baseline
- the initial context-aware stacker overfit severely, motivating the regularized meta-model update now included in the repository

This makes the project scientifically stronger: the repository does not just report successes, it also documents failure modes and mitigation strategies for ensemble overfitting.

## Why Different Models Matter

- **ARIMAX** captures smoother linear structure and interpretable relationships with exogenous signals.
- **XGBoost** captures nonlinear cross-feature effects without heavy feature scaling demands.
- **LSTM** captures sequential dependencies and temporal context across rolling windows.
- **Context-aware stacking** combines them by learning which model to trust under different market conditions instead of assigning fixed weights.

## ESE 588 Alignment

This repository is now directly aligned with the ESE 588 course project expectations:

- clear mathematical problem formulation
- explicit method section with model choices and design rationale
- **synthetic experiments** to study model behavior under controlled conditions
- **real-data experiments** on SPY
- discussion of strengths, limitations, and failure cases

The strongest course-report framing is:

- Does adaptive ensemble learning improve 5-day stock direction forecasting?
- Under what regimes do ARIMAX, XGBoost, and LSTM succeed or fail?
- Can context-aware stacking generalize across changing market conditions?

## Future Work

- Extend from single-ticker SPY modeling to multi-asset panels.
- Add walk-forward retraining instead of one-shot fitting.
- Introduce probability calibration and threshold optimization.
- Add richer macroeconomic and cross-asset exogenous features.
- Replace single validation-period stacking with out-of-fold temporal meta-features.
- Add model registry, experiment tracking, and automated backtest reports.
<<<<<<< HEAD

>>>>>>> 28e1cb3 (Initial stock forecasting project scaffold)
=======
>>>>>>> 26f60e1 (   README.md)
