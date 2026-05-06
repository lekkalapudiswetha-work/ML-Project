"""ARIMAX model selection and forecasting utilities."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation import compute_classification_metrics


ARIMAX_ORDERS = [(1, 1, 1), (2, 1, 2), (3, 1, 1)]


def _fit_single_arimax(
    train_target: pd.Series,
    train_exog: pd.DataFrame,
    order: tuple[int, int, int],
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_target,
            exog=train_exog,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
    return result


def train_arimax_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exog_columns: list[str],
) -> dict[str, object]:
    """Fit candidate ARIMAX models, select by validation accuracy then AIC."""
    candidates: list[dict[str, object]] = []
    train_target = train_df["future_return"]
    train_exog = train_df[exog_columns]
    validation_exog = validation_df[exog_columns]

    for order in ARIMAX_ORDERS:
        result = _fit_single_arimax(train_target, train_exog, order)
        validation_forecast = result.get_forecast(
            steps=len(validation_df),
            exog=validation_exog,
        ).predicted_mean
        validation_pred = (validation_forecast.values > 0).astype(int)
        validation_metrics = compute_classification_metrics(validation_df["target"], validation_pred)
        candidates.append(
            {
                "order": order,
                "result": result,
                "aic": result.aic,
                "validation_forecast": validation_forecast.values,
                "validation_pred": validation_pred,
                "validation_metrics": validation_metrics,
            }
        )

    best = sorted(
        candidates,
        key=lambda item: (-item["validation_metrics"]["accuracy"], item["aic"]),
    )[0]

    validation_pred = best["validation_pred"]
    validation_prob = 1 / (1 + np.exp(-best["validation_forecast"]))

    test_forecast = best["result"].get_forecast(
        steps=len(test_df),
        exog=test_df[exog_columns],
    ).predicted_mean.values
    test_pred = (test_forecast > 0).astype(int)
    test_prob = 1 / (1 + np.exp(-test_forecast))

    return {
        "model": best["result"],
        "selected_order": best["order"],
        "candidate_summary": pd.DataFrame(
            [
                {
                    "order": str(candidate["order"]),
                    "aic": candidate["aic"],
                    "validation_accuracy": candidate["validation_metrics"]["accuracy"],
                }
                for candidate in candidates
            ]
        ),
        "validation_predictions": validation_pred,
        "validation_probabilities": validation_prob,
        "validation_metrics": best["validation_metrics"],
        "test_predictions": test_pred,
        "test_probabilities": test_prob,
        "test_metrics": compute_classification_metrics(test_df["target"], test_pred),
    }

