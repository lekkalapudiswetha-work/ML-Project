# ESE 588 Project Report Draft

## Adaptive Ensemble Stock Forecasting with Context-Aware Stacking

### 1. Introduction

Forecasting financial markets is a difficult machine learning problem because the data are noisy, nonstationary, and affected by changing market regimes. Even when predictive signals exist, they are often weak and unstable over time. As a result, no single model family is guaranteed to perform well across all market conditions.

This project studies a short-horizon financial prediction problem: determining whether the price of a security will be higher after five trading days. The main real-data experiment focuses on SPY, the SPDR S&P 500 ETF Trust. SPY was chosen because it is smoother and less idiosyncratic than many individual stocks, which makes it a useful first benchmark for an ensemble forecasting system. The five-day horizon was selected because one-day prediction is often too noisy, while much longer horizons can mix short-term technical structure with broader macro trends.

The main motivation of this project is that different model classes capture different types of structure. ARIMAX is appropriate when the relationships are relatively linear and interpretable. XGBoost is suitable when nonlinear interactions among features matter. LSTM is designed to exploit temporal dependence through rolling sequential inputs. Instead of treating these models only as competing alternatives, this project investigates whether they can be combined through stacking, and more specifically through a context-aware stacking approach that uses market regime information such as volatility, trend, and momentum.

The goal is not only to maximize prediction performance, but also to understand when and why different models work. To support that goal, the project includes both synthetic experiments and real-data experiments. The synthetic experiments test the models under controlled linear, nonlinear, sequential, and regime-switching processes. The real-data experiment evaluates whether the same conclusions hold on historical SPY data.

### 2. Problem Formulation

Let P_t denote the adjusted closing price of the asset at trading day t.

The primary target is a binary classification label:

    y_t = 1 if P_(t+5) > P_t, else 0

This means the model predicts whether the asset price will be higher after five trading days.

The project also defines a continuous forward return target:

    r_t_5 = (P_(t+5) / P_t) - 1

This forward return is used by the ARIMAX model, which naturally produces continuous forecasts.

The supervised learning task is:

    f(x_t) -> p_hat_t

where p_hat_t is the predicted probability that y_t = 1 and x_t is the feature vector available at time t.

The final class prediction is:

    y_hat_t = 1 if p_hat_t >= 0.5, else 0

The feature vector includes:

- Moving averages: MA5, MA20, MA50
- Trend signal: MA5 - MA20
- Moving average crossover indicator
- Rolling volatility over 5, 10, and 20 days
- RSI
- MACD
- Momentum over 5 days
- Daily return
- Lagged returns from lag_1 to lag_5
- Volume moving average
- Volume percentage change

The main assumptions are:

- Historical price and volume signals contain weak but useful predictive information.
- Different models capture different types of structure.
- Model performance depends on market regime.
- Chronological splitting is necessary to avoid temporal leakage.

For the real-data experiment, the chronological split is:

- Train: 2013-01-01 to 2021-12-31
- Validation: 2022-01-01 to 2022-12-31
- Test: 2023-01-01 to 2024-01-01

The training set is used to fit base models. The validation set is used for model comparison and to generate meta-features for stacking. The test set is used only for final evaluation.

### 3. Method

The project implements three base learners and two ensemble stackers.

#### 3.1 ARIMAX

The first base learner is ARIMAX, implemented using SARIMAX from statsmodels.

The model predicts the 5-day forward return:

    r_t_5 = g(past_returns, exogenous_features) + error

The exogenous variables are:

- trend_signal
- vol10
- rsi

Three candidate orders are tested:

- (1, 1, 1)
- (2, 1, 2)
- (3, 1, 1)

The model is selected based on validation accuracy, with AIC used as a secondary criterion. Since ARIMAX produces a continuous forecast, the binary prediction is:

    y_hat_ARIMAX_t = 1 if r_hat_t_5 > 0, else 0

A pseudo-probability is produced by applying a logistic transformation to the forecasted return.

ARIMAX provides an interpretable baseline and is most appropriate when the underlying process is smooth and approximately linear.

#### 3.2 XGBoost

The second base learner is XGBoost, implemented using XGBClassifier.

The model is trained on the full engineered feature set. The main configuration includes:

- 250 trees
- max_depth = 4
- learning_rate = 0.05
- subsample = 0.9
- colsample_bytree = 0.9

XGBoost is useful when nonlinear relationships and feature interactions matter. Unlike ARIMAX, it does not assume a linear time-series structure. Instead, it learns a nonlinear decision boundary directly from the engineered features.

#### 3.3 LSTM

The third base learner is an LSTM model implemented in TensorFlow/Keras.

The LSTM uses rolling 20-day input windows. The sequential input features are:

- trend_signal
- vol10
- rsi
- lag_1
- lag_2
- lag_3
- lag_4
- lag_5

The architecture is:

- Input layer
- LSTM with 32 hidden units
- Dropout with rate 0.2
- Dense sigmoid output

The model uses:

- Adam optimizer
- Binary cross-entropy loss
- 20 epochs
- Batch size 32

Features are scaled with MinMaxScaler fitted only on the training set. The LSTM is intended to capture temporal structure that static tabular models may miss.

#### 3.4 Stacking and Context-Aware Stacking

Two stackers are implemented.

The first is logistic regression stacking. It uses only the base model outputs:

- xgb_prob
- lstm_prob
- arimax_prob

This is a static ensemble because it does not use additional market context.

The second is context-aware XGBoost stacking. It uses both base predictions and context features:

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

The motivation is that model reliability may change across regimes. For example:

- LSTM may work better when temporal state persistence is strong.
- ARIMAX may work better in smoother linear regimes.
- XGBoost may work better when nonlinear interactions are important.

The initial context-aware stacker overfit the validation set very strongly. To reduce this, the final implementation includes:

- smaller trees
- stronger L1 and L2 regularization
- smaller subsampling ratios
- chronological holdout split inside the validation period
- early stopping

This reduced the validation-test gap, although it did not make the context-aware stacker the best held-out model.

#### 3.5 Evaluation

The project uses both classification metrics and financial metrics.

Classification metrics:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- Classification report

Financial evaluation:

- Cumulative return
- Sharpe ratio
- Long/flat trading rule

Trading logic:

- Go long if prediction = 1
- Stay flat if prediction = 0

This makes it possible to evaluate whether the model is useful not only as a classifier but also as a simple trading signal.

### 4. Experiments on Synthetic Data

The synthetic experiments were designed to test whether different models behave as expected under controlled data-generating regimes. This is important because real market data contain many confounding effects and do not clearly isolate model inductive bias.

Synthetic returns are first generated according to regime-specific equations, and then converted to prices using:

    P_t = P_(t-1) * (1 + r_t)

Synthetic volume is generated as a function of return magnitude plus noise so that the same real-data feature-engineering pipeline can be reused.

Each synthetic dataset is split chronologically into train, validation, and test segments. The same models used on SPY are then trained and evaluated on these synthetic datasets.

#### 4.1 Linear Regime

The first synthetic regime is a linear autoregressive process with an exogenous variable:

    r_t = 0.25 * r_(t-1) + 0.015 * z_t + error_t

where z_t is Gaussian noise and error_t has adjustable variance.

This regime was intended to favor ARIMAX.

Observed results:

- At low noise, context-aware stacking gave the best test F1: 0.5320
- ARIMAX test F1: 0.2616
- XGBoost test F1: 0.0194
- LSTM test F1: 0.0000

- At medium noise, logistic stacking gave the best test F1: 0.6746
- Context-aware stacking test F1: 0.6617
- ARIMAX test F1: 0.1433

- At high noise, both stackers again performed best
- Context-aware stacking test F1: 0.7160
- Logistic stacking test F1: 0.7119
- LSTM test F1: 0.5063
- ARIMAX test F1: 0.1970

Interpretation:

Although the return process was linear, ARIMAX did not dominate the final 5-day classification task. This suggests that the transformation from returns to engineered technical features and multi-day directional labels introduces complexity beyond the ARIMAX decision boundary.

#### 4.2 Nonlinear Threshold Regime

The nonlinear regime introduces threshold behavior where returns depend on the sign of a latent signal. This regime was intended to favor XGBoost.

Observed results:

- Low noise:
  - XGBoost test F1: 0.6752
  - Logistic stacking test F1: 0.6604
  - ARIMAX test F1: 0.5899

- Medium noise:
  - Logistic stacking test F1: 0.6720
  - ARIMAX test F1: 0.6336
  - XGBoost test F1: 0.5643

- High noise:
  - ARIMAX test F1: 0.6185
  - Logistic stacking test F1: 0.5735
  - XGBoost test F1: 0.5426

Interpretation:

XGBoost is competitive in nonlinear settings, especially at low noise, but the ensemble often benefits from combining complementary information across multiple base learners. Logistic stacking frequently outperformed the individual models. Context-aware stacking did not consistently dominate.

#### 4.3 Sequential Latent-State Regime

The sequential regime introduces persistent hidden states and temporal memory. This regime was designed to favor the LSTM.

Observed results:

- Low noise:
  - LSTM test accuracy: 0.6592
  - LSTM test F1: 0.7319
  - ARIMAX test F1: 0.5403
  - XGBoost test F1: 0.0369

- Medium noise:
  - LSTM test F1: 0.7102
  - Context-aware stacking test F1: 0.7091
  - Logistic stacking test F1: 0.6840

- High noise:
  - Logistic stacking test F1: 0.6965
  - Context-aware stacking test F1: 0.6846
  - ARIMAX test F1: 0.6064
  - LSTM test F1: 0.5843

Interpretation:

This regime gave the clearest support for the LSTM inductive bias. When temporal dependence is strong and noise is moderate, LSTM is the strongest base learner. At higher noise levels, the ensemble becomes more useful because the sequence signal becomes less clean.

#### 4.4 Regime-Switching Regime

The regime-switching process combines a low-volatility linear state with a higher-volatility nonlinear state. This is the most important synthetic setting because it tests the main hypothesis behind context-aware stacking.

Observed results:

- Low noise:
  - LSTM test F1: 0.6945
  - Logistic stacking test F1: 0.6870
  - Context-aware stacking test F1: 0.6288

- Medium noise:
  - Logistic stacking test F1: 0.6851
  - LSTM test F1: 0.6526
  - Context-aware stacking test F1: 0.5386

- High noise:
  - LSTM test F1: 0.6859
  - Logistic stacking test F1: 0.6835
  - Context-aware stacking test F1: 0.6835

Sample size study:

- Small sample:
  - LSTM test F1: 0.8000
  - Logistic stacking test F1: 0.7029
  - Context-aware stacking test F1: 0.4933

- Medium sample:
  - ARIMAX test F1: 0.6450
  - XGBoost test F1: 0.5562
  - Context-aware stacking test F1: 0.0000

- Large sample:
  - ARIMAX test F1: 0.6366
  - Logistic stacking test F1: 0.6708
  - Context-aware stacking test F1: 0.6708

Interpretation:

The context-aware stacker often performed strongly on validation data, but its test performance was inconsistent. This closely mirrors the overfitting behavior observed on real SPY data. The main conclusion is that context-aware meta-learning is itself sensitive to regime composition, sample size, and regularization.

#### 4.5 Synthetic Experiment Summary

The synthetic experiments support the following conclusions:

- LSTM is strongest when sequential memory matters.
- XGBoost is useful when nonlinear threshold structure matters.
- ARIMAX remains a meaningful linear baseline.
- Stacking often improves over individual base models.
- Context-aware stacking is promising but less stable than simpler stacking.

These experiments provide a controlled explanation for the real-data behavior observed on SPY.

### 5. Experiments on Real Data

#### 5.1 Dataset

The real-data experiments use SPY, the SPDR S&P 500 ETF Trust, from 2013-01-01 to 2024-01-01. The data are downloaded using yfinance. Adjusted close is used as the main price series, and volume is included in feature engineering.

The preprocessing pipeline:

- Flattens any MultiIndex columns returned by yfinance
- Uses adjusted close as the main close series
- Creates a datetime index
- Sorts chronologically
- Handles missing values before feature construction

#### 5.2 Experimental Setup

Chronological split:

- Train: 2013-01-01 to 2021-12-31
- Validation: 2022-01-01 to 2022-12-31
- Test: 2023-01-01 to 2024-01-01

Procedure:

1. Train ARIMAX, XGBoost, and LSTM on the train set
2. Generate validation predictions from the base models
3. Train logistic stacking and context-aware stacking on validation predictions
4. Evaluate all models on the test set

#### 5.3 Real-Data Results

Validation results:

- XGBoost:
  - Accuracy: 0.4900
  - F1: 0.6343

- ARIMAX:
  - Accuracy: 0.5020
  - F1: 0.5211

- LSTM:
  - Accuracy: 0.4661
  - F1: 0.6359

- Logistic Stacking:
  - Accuracy: 0.5259
  - F1: 0.2699

- Context-Aware XGBoost Stacking:
  - Accuracy: 0.6813
  - F1: 0.6154

Test results:

- XGBoost:
  - Accuracy: 0.5388
  - Precision: 0.6039
  - Recall: 0.8013
  - F1: 0.6887
  - Sharpe: 3.2873
  - Cumulative return: 1.3942

- ARIMAX:
  - Accuracy: 0.4286
  - Precision: 0.5889
  - Recall: 0.3397
  - F1: 0.4309
  - Sharpe: 2.2593
  - Cumulative return: 0.5494

- LSTM:
  - Accuracy: 0.6367
  - Precision: 0.6367
  - Recall: 1.0000
  - F1: 0.7781
  - Sharpe: 4.1685
  - Cumulative return: 2.1429

- Logistic Stacking:
  - Accuracy: 0.4367
  - Precision: 0.8000
  - Recall: 0.1538
  - F1: 0.2581
  - Sharpe: 2.7956
  - Cumulative return: 0.4873

- Context-Aware XGBoost Stacking:
  - Accuracy: 0.3918
  - Precision: 0.7692
  - Recall: 0.0641
  - F1: 0.1183
  - Sharpe: 2.4827
  - Cumulative return: 0.2975

#### 5.4 Interpretation

The LSTM is the strongest held-out test model. This suggests that short-horizon directional information in SPY is better captured through temporal sequence modeling than through a purely linear or purely tabular approach.

XGBoost is also competitive, which indicates that the technical feature space contains useful nonlinear structure.

ARIMAX is weaker as a classifier, but still provides positive trading statistics. This is important because classification quality and economic usefulness are not always identical.

The stackers perform poorly on test data relative to the best base learners. The context-aware stacker originally overfit the validation set extremely strongly. After regularization, the validation-test gap decreased substantially, but the model still did not outperform LSTM or XGBoost on test data.

An important detail is that the context-aware stacker has high precision but extremely low recall. This means it makes very few positive predictions, but many of those predictions are correct. In practice, this behaves like a sparse and conservative signal rather than a balanced classifier.

### 6. Discussion

The main strength of the project is that it connects modeling assumptions with controlled and real-world evaluation. The synthetic experiments make it possible to test whether each model behaves as expected under known structure. The SPY experiment then shows how those same ideas behave in a realistic financial dataset.

Another strength is the diversity of model classes. ARIMAX, XGBoost, and LSTM are not minor variants of one another. They represent genuinely different approaches:

- ARIMAX for structured linear dynamics
- XGBoost for nonlinear feature interaction
- LSTM for sequential memory

The ensemble framework is also meaningful because it does not assume a single correct inductive bias.

The project also uses an appropriate time-series evaluation design. Chronological splitting is essential in forecasting, and avoiding random train-test splits makes the results much more credible.

The biggest limitation is the instability of context-aware stacking. Both synthetic and real-data experiments show that the adaptive stacker can fit validation regimes strongly but generalize poorly. This indicates that the meta-learning stage is itself a difficult forecasting problem.

Other limitations include:

- The real-data experiment is limited to SPY
- The feature set is purely technical and volume-based
- The backtest ignores transaction costs, slippage, and position sizing
- The stackers are trained on relatively limited meta-data

The most important scientific insight is that adaptive stacking is not automatically superior. It can help when model reliability truly changes across regimes, but it also introduces a second layer of overfitting risk. This is one of the central lessons of the project.

### 7. Conclusion

This project investigated five-day stock direction forecasting using a heterogeneous ensemble framework built from ARIMAX, XGBoost, LSTM, logistic stacking, and context-aware stacking.

The synthetic experiments confirmed that model inductive bias matters:

- LSTM is strongest in sequential latent-state regimes
- XGBoost is effective in nonlinear regimes
- ARIMAX is a meaningful linear baseline
- Stacking can improve performance by combining complementary structure

However, the synthetic experiments also showed that context-aware stacking is less stable than simpler stacking under limited sample sizes or unstable regime composition.

The real-data experiment on SPY showed that LSTM is the strongest held-out model, with XGBoost as a competitive second-best model. ARIMAX remains a useful baseline. The context-aware stacker, even after regularization, did not outperform the best base learners and remained sensitive to overfitting.

The final conclusion is therefore nuanced. Adaptive ensemble learning is promising for financial forecasting, but its success depends heavily on how the meta-learning stage is constructed and regularized. In this project, model diversity clearly helped reveal different types of structure, but robust context-aware stacking remained challenging in both synthetic and real settings.

### References

Suggested references for the final submitted report:

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. Time Series Analysis: Forecasting and Control.
2. Chen, T., and Guestrin, C. XGBoost: A Scalable Tree Boosting System.
3. Hochreiter, S., and Schmidhuber, J. Long Short-Term Memory.
4. Wolpert, D. Stacked Generalization.
5. Relevant documentation for statsmodels, xgboost, tensorflow, and yfinance.
