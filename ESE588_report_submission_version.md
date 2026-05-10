# Adaptive Ensemble Stock Forecasting with Context-Aware Stacking

## 1. Introduction

Financial forecasting is a challenging machine learning problem because market data are noisy, nonstationary, and influenced by changing regimes. Even when predictive structure exists, it is often weak and unstable over time. This makes it unlikely that any single model class will perform well across all conditions.

This project studies a short-horizon prediction task: determining whether the price of a security will be higher after five trading days. The main real-data experiment focuses on SPY, the SPDR S&P 500 ETF Trust. SPY was chosen because it is smoother than many individual equities and therefore provides a stable first benchmark for an ensemble forecasting system. The five-day horizon was selected because one-day prediction is often excessively noisy, while much longer horizons mix short-term technical effects with broader market drift.

The central hypothesis of this project is that different model classes capture different types of structure. ARIMAX is appropriate for smooth linear dependence, XGBoost is appropriate for nonlinear tabular interactions, and LSTM is appropriate for temporal dependence across rolling windows. Instead of treating these models only as competitors, this project investigates whether they can be combined through stacking, and more specifically through a context-aware stacking mechanism that conditions on market regime features such as volatility, momentum, and trend.

To evaluate this hypothesis rigorously, the project uses both synthetic and real data. Synthetic experiments are used to test model behavior under controlled linear, nonlinear, sequential, and regime-switching settings. Real-data experiments on SPY are then used to determine whether those conclusions remain meaningful in historical market data.

## 2. Problem Formulation

Let P_t denote the adjusted closing price at trading day t. The primary target is a binary directional label:

    y_t = 1 if P_(t+5) > P_t, else 0

This corresponds to predicting whether the price will be higher after five trading days.

The project also defines a forward return target:

    r_t_5 = (P_(t+5) / P_t) - 1

This continuous target is used by the ARIMAX model.

The supervised learning problem is to learn a mapping

    f(x_t) -> p_hat_t

where x_t is the feature vector available at time t and p_hat_t is the predicted probability that y_t = 1. The final prediction rule is:

    y_hat_t = 1 if p_hat_t >= 0.5, else 0

The feature vector includes trend, volatility, momentum, lagged returns, and volume-based indicators:

- MA5, MA20, MA50
- trend_signal = MA5 - MA20
- moving average crossover indicator
- vol5, vol10, vol20
- RSI
- MACD
- momentum5
- daily return
- lag_1 through lag_5
- volume_ma20
- volume_change

The main assumptions are:

1. historical price and volume information contain weak but useful predictive structure;
2. model performance is regime-dependent;
3. different model classes capture different forms of dependence;
4. chronological splitting is necessary to avoid temporal leakage.

For the real-data experiment, the chronological split is:

- Train: 2013-01-01 to 2021-12-31
- Validation: 2022-01-01 to 2022-12-31
- Test: 2023-01-01 to 2024-01-01

This design ensures that model fitting, model selection, and final evaluation are separated in time.

## 3. Method

The project implements three base learners and two stacking models.

### 3.1 ARIMAX

The first base learner is ARIMAX, implemented using SARIMAX from statsmodels. It predicts the 5-day forward return as a function of past return structure and exogenous signals:

    r_t_5 = g(past_returns, exogenous_features) + error_t

The exogenous variables are:

- trend_signal
- vol10
- rsi

Three candidate orders are evaluated:

- (1, 1, 1)
- (2, 1, 2)
- (3, 1, 1)

The final model is selected using validation accuracy, with AIC used as a secondary criterion. The directional prediction is produced by thresholding the forecasted return:

    y_hat_ARIMAX_t = 1 if r_hat_t_5 > 0, else 0

ARIMAX provides an interpretable econometric baseline.

### 3.2 XGBoost

The second base learner is XGBoost, implemented using XGBClassifier. The model is trained on the full engineered feature set and uses the following main settings:

- 250 trees
- max_depth = 4
- learning_rate = 0.05
- subsample = 0.9
- colsample_bytree = 0.9

XGBoost is designed to capture nonlinear interactions among technical indicators and lagged signals without imposing a linear structure.

### 3.3 LSTM

The third base learner is an LSTM implemented in TensorFlow/Keras. The model uses 20-day rolling windows with the following inputs:

- trend_signal
- vol10
- rsi
- lag_1 to lag_5

The architecture is:

- Input
- LSTM(32)
- Dropout(0.2)
- Dense(1, sigmoid)

Features are scaled with MinMaxScaler fitted only on the training set. The LSTM is intended to capture sequential dependence that static feature models may miss.

### 3.4 Stacking and Context-Aware Stacking

Two stackers are implemented.

The first is logistic regression stacking using only base model outputs:

- xgb_prob
- lstm_prob
- arimax_prob

The second is context-aware XGBoost stacking using both base outputs and context variables:

- xgb_prob
- lstm_prob
- arimax_prob
- vol10
- vol20
- rsi
- trend_signal
- momentum5
- macd
- volume_change

The motivation is that model reliability is conditional on market regime. However, the initial context-aware stacker overfit strongly. To mitigate this, the final implementation includes:

- shallower trees
- fewer boosting rounds
- stronger L1 and L2 regularization
- reduced row and column subsampling
- chronological holdout inside validation
- early stopping

### 3.5 Evaluation

The project uses both classification and financial evaluation.

Classification metrics:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

Financial metrics:

- cumulative return
- Sharpe ratio

The trading rule is long/flat:

- long if prediction = 1
- flat if prediction = 0

[Table 1 here: Summary of models, inputs, and roles]

## 4. Experiments on Synthetic Data

Synthetic experiments were designed to test whether model performance aligns with the structure of the data-generating process. Synthetic returns were generated first, then converted to price paths using:

    P_t = P_(t-1) * (1 + r_t)

Synthetic volume was also generated so that the same feature-engineering pipeline used for SPY could be reused.

Each synthetic dataset was split chronologically into training, validation, and test partitions. Four regimes were studied:

1. linear autoregressive regime
2. nonlinear threshold regime
3. sequential latent-state regime
4. regime-switching regime

Noise levels were varied across low, medium, and high settings. A sample-size sensitivity study was also performed for the regime-switching case.

### 4.1 Linear Regime

The linear regime followed:

    r_t = 0.25 * r_(t-1) + 0.015 * z_t + error_t

This setting was intended to favor ARIMAX. However, ARIMAX did not dominate the final 5-day classification task. At medium noise, ARIMAX achieved test F1 of 0.1433, while logistic stacking and context-aware stacking achieved test F1 values of 0.6746 and 0.6617, respectively. This suggests that even when the return process is linear, the transformed classification problem based on technical indicators is more complex than the original return equation.

### 4.2 Nonlinear Threshold Regime

The nonlinear regime introduced threshold-dependent behavior. This setting was intended to favor XGBoost. At low noise, XGBoost performed strongly with test F1 of 0.6752, but logistic stacking was often equally strong or stronger across settings. At medium noise, logistic stacking achieved test F1 of 0.6720, outperforming XGBoost and context-aware stacking. These results show that nonlinear structure benefits from model combination, not only from a single nonlinear learner.

### 4.3 Sequential Latent-State Regime

The sequential regime introduced persistent hidden states and autoregressive memory. This was the clearest success case for the LSTM. At low noise, LSTM achieved test accuracy of 0.6592 and test F1 of 0.7319, substantially outperforming ARIMAX and XGBoost. At medium noise, LSTM remained one of the strongest models. This confirms that recurrent sequence models are most useful when predictive information depends on temporal persistence rather than only on static feature interactions.

### 4.4 Regime-Switching Regime

The regime-switching process combined a low-volatility linear state with a higher-volatility nonlinear state. This was the main test for context-aware stacking. Validation performance for the context-aware stacker was often strong, but test performance was inconsistent. At medium noise, context-aware stacking reached test F1 of 0.5386, while logistic stacking and LSTM reached 0.6851 and 0.6526. In the sample-size study, context-aware stacking was unstable, including collapse to zero F1 in one medium-sample setting. This mirrors the real-data overfitting problem and suggests that adaptive meta-learning is highly sensitive to data size and regime composition.

### 4.5 Synthetic Summary

The synthetic experiments support four conclusions:

1. LSTM is strongest when temporal memory matters.
2. XGBoost is useful when threshold-type nonlinearity matters.
3. ARIMAX is a meaningful linear baseline but is not always best for the final classification task.
4. Context-aware stacking is promising in theory but less stable than simpler stacking.

[Table 2 here: Medium-noise synthetic test results across regimes]

## 5. Experiments on Real Data

### 5.1 Dataset

The real-data experiment uses SPY from 2013-01-01 to 2024-01-01, downloaded using yfinance. Adjusted close is used as the main price series and volume is used for additional features. The data are chronologically sorted and cleaned before feature construction.

### 5.2 Experimental Setup

The chronological split is:

- Train: 2013-01-01 to 2021-12-31
- Validation: 2022-01-01 to 2022-12-31
- Test: 2023-01-01 to 2024-01-01

Base models are trained on the train set. Validation predictions are then used to train the stackers. The test set is reserved strictly for final evaluation.

### 5.3 Results

Validation results:

- XGBoost: accuracy 0.4900, F1 0.6343
- ARIMAX: accuracy 0.5020, F1 0.5211
- LSTM: accuracy 0.4661, F1 0.6359
- Logistic Stacking: accuracy 0.5259, F1 0.2699
- Context-Aware XGBoost Stacking: accuracy 0.6813, F1 0.6154

Test results:

- XGBoost: accuracy 0.5388, F1 0.6887, Sharpe 3.2873, cumulative return 1.3942
- ARIMAX: accuracy 0.4286, F1 0.4309, Sharpe 2.2593, cumulative return 0.5494
- LSTM: accuracy 0.6367, F1 0.7781, Sharpe 4.1685, cumulative return 2.1429
- Logistic Stacking: accuracy 0.4367, F1 0.2581, Sharpe 2.7956, cumulative return 0.4873
- Context-Aware XGBoost Stacking: accuracy 0.3918, F1 0.1183, Sharpe 2.4827, cumulative return 0.2975

### 5.4 Interpretation

LSTM is the strongest held-out model on SPY, both in classification and financial terms. This indicates that short-horizon directional information in SPY is more effectively captured through temporal sequence modeling than through either a linear econometric model or a static tabular learner alone.

XGBoost is also competitive, confirming that the technical indicator feature space contains useful nonlinear structure. ARIMAX is weaker in terms of classification performance but still produces positive trading statistics, which highlights that financial usefulness and classification accuracy are not always identical.

The stackers do not outperform the strongest base learners on test data. The context-aware stacker originally overfit validation data severely; regularization reduced this problem but did not make it the best final model. Its high test precision and extremely low recall show that it produces sparse conservative signals rather than balanced directional forecasts.

[Table 3 here: Real-data validation and test results]

## 6. Discussion

The strongest aspect of this project is the connection between theory and empirical evaluation. The synthetic experiments isolate model inductive bias under known structure, while the SPY experiment tests whether the same conclusions remain meaningful in real market data.

The results show that no single model class is universally best. Instead, model performance depends on the structure of the data-generating process:

- LSTM performs best when predictive structure is sequential.
- XGBoost is effective for nonlinear interactions.
- ARIMAX provides an interpretable linear baseline.
- Stacking can improve performance when models capture complementary signals.

At the same time, the project shows that context-aware stacking is difficult to train robustly. Both synthetic and real-data experiments show that the adaptive stacker can fit validation regimes strongly but generalize inconsistently. This indicates that the meta-learning stage is itself sensitive to sample size, regime definition, and regularization.

The project has several limitations. First, the real-data analysis focuses only on SPY, so conclusions may not generalize directly to more volatile individual stocks. Second, the feature set is restricted to technical and volume-based variables. Third, the financial backtest ignores transaction costs, slippage, and risk constraints. Finally, the meta-learning framework would likely benefit from more robust walk-forward or out-of-fold temporal training rather than a single validation-year approach.

Despite these limitations, the project provides a meaningful result: adaptive stacking is not automatically superior, and honest evaluation of failure cases is essential. This is a useful conclusion in financial machine learning, where overly flexible models can appear strong in-sample while failing under regime shift.

## 7. Conclusion

This project investigated five-day stock direction forecasting using a heterogeneous ensemble framework composed of ARIMAX, XGBoost, LSTM, logistic stacking, and context-aware stacking.

The synthetic experiments showed that model inductive bias matters. LSTM performed best in sequential latent-state regimes, XGBoost was effective in nonlinear settings, and ARIMAX remained a useful linear baseline. Stacking often improved over single-model performance, but context-aware stacking was less stable than simpler stacking.

The real-data experiments on SPY showed that LSTM was the strongest held-out model, with XGBoost as a competitive alternative and ARIMAX as an interpretable benchmark. The context-aware stacker, even after regularization, did not outperform the best base learner and remained sensitive to overfitting.

The final conclusion is therefore mixed but informative. Adaptive ensemble learning is promising for financial forecasting, but its success depends strongly on the reliability of the meta-learning stage. In this project, model diversity clearly helped reveal different types of predictive structure, but robust context-aware stacking remained challenging in both synthetic and real settings.

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. Time Series Analysis: Forecasting and Control.
2. Chen, T., and Guestrin, C. XGBoost: A Scalable Tree Boosting System.
3. Hochreiter, S., and Schmidhuber, J. Long Short-Term Memory.
4. Wolpert, D. Stacked Generalization.
5. Documentation for statsmodels, xgboost, tensorflow, and yfinance.
