Adaptive Ensemble Stock Forecasting with Context-Aware Stacking

Swetha Lekkalapudi
ESE 588
May 10, 2026

1. Introduction

Financial forecasting remains one of the most challenging applications of machine learning. Unlike many benchmark datasets, financial time series are characterized by low signal-to-noise ratios, nonstationarity, structural breaks, and frequent regime changes. A model that performs well during one period can deteriorate rapidly when volatility, trend behavior, or market microstructure changes. This makes it difficult to rely on any single modeling family across all conditions.

This project studies a short-horizon forecasting problem: predicting whether the price of a security will be higher after five trading days. The real-data component focuses on SPY, the SPDR S&P 500 ETF Trust, which tracks the S&P 500 index. SPY was selected as the initial benchmark because it is smoother and less idiosyncratic than many individual equities, making it a useful starting point for evaluating an ensemble forecasting pipeline. The five-day horizon was chosen because one-day forecasting is often dominated by noise, while much longer horizons can mix short-term technical dynamics with broader macroeconomic drift.

The motivation for this project is that different models capture different forms of predictive structure. Classical econometric models such as ARIMAX are useful when the underlying dynamics are approximately linear and interpretable. Tree-based models such as XGBoost are effective when directional behavior depends on nonlinear interactions among engineered indicators. Sequence models such as LSTM are better suited for persistent temporal structure spread over rolling windows. Rather than assuming that one of these models should dominate universally, this project investigates whether they can be combined productively through stacking, and more specifically through a context-aware stacking strategy that conditions ensemble behavior on market regime indicators.

The core research question is not simply whether an ensemble can outperform every single model on a real dataset. A more meaningful question is: under what conditions does adaptive ensemble learning help, and when does the meta-learning layer become unstable or overfit? To answer that question, the project uses both synthetic and real-data experiments. Synthetic experiments test model behavior under controlled linear, nonlinear, sequential, and regime-switching settings. Real-data experiments on SPY evaluate how those conclusions transfer to a noisy historical market series. This design allows the project to connect modeling assumptions, empirical behavior, and failure modes in a way that aligns well with the goals of graduate-level machine learning analysis.

2. Problem Formulation

Let P_t denote the adjusted closing price at trading day t. The primary target is a binary directional label defined by

y_t = 1 if P_(t+5) > P_t, else 0

This label indicates whether the asset price is higher after five trading days. The problem is therefore framed as a binary classification task.

In addition to the classification target, the project defines a continuous five-day forward return:

r_t_5 = (P_(t+5) / P_t) - 1

This continuous target is used by ARIMAX, which naturally produces return forecasts rather than class probabilities.

The supervised learning task is to estimate a mapping

f(x_t) -> p_hat_t

where x_t is the feature vector available at time t and p_hat_t is the model-estimated probability that y_t = 1. The final prediction rule is

y_hat_t = 1 if p_hat_t >= threshold, else 0

For most models the threshold is 0.5, although the final context-aware stacker uses a threshold selected on validation data.

The engineered features are based entirely on historical price and volume information. They include:
MA5, MA20, MA50
trend_signal = MA5 - MA20
moving average crossover indicator
vol5, vol10, vol20
RSI
MACD
momentum5
daily return
lag_1 through lag_5
volume_ma20
volume_change

These features were chosen to represent trend, volatility, momentum, short-term return dynamics, and trading activity.

The project assumes:
1. Historical price and volume contain weak but useful information about future five-day direction.
2. Predictive relationships are regime-dependent rather than stationary.
3. Different model classes capture different aspects of the forecasting problem.
4. Strict chronological data splitting is necessary to avoid temporal leakage.

The real-data experiment uses the following chronological split:
Train: 2013-01-01 to 2021-12-31
Validation: 2022-01-01 to 2022-12-31
Test: 2023-01-01 to 2024-01-01

This structure ensures that the base models are trained only on past data, the meta-models are trained only on validation predictions, and the test set remains an unbiased final evaluation set.

3. Method

The project implements three base learners and two stacking models. These models were chosen to represent distinct inductive biases rather than minor variations of the same framework.

3.1 ARIMAX

The first base learner is ARIMAX, implemented using SARIMAX from statsmodels. It models the five-day forward return using autoregressive structure and exogenous signals:

r_t_5 = g(past_returns, exogenous_features) + error_t

The exogenous variables used are:
trend_signal
vol10
rsi

Three candidate orders were evaluated:
(1, 1, 1)
(2, 1, 2)
(3, 1, 1)

The final ARIMAX specification was chosen using validation accuracy, with AIC used as a secondary model-selection criterion. Since ARIMAX produces a continuous return forecast, its directional output is obtained by thresholding:

y_hat_ARIMAX_t = 1 if r_hat_t_5 > 0, else 0

ARIMAX serves as the classical interpretable baseline. It is especially useful for understanding whether the predictive structure is close to linear and whether a simpler econometric model is already sufficient.

3.2 XGBoost

The second base learner is XGBoost, implemented as an XGBClassifier. The model is trained on the full engineered feature set and uses 250 trees, maximum depth 4, learning rate 0.05, and row and column subsampling. XGBoost is intended to capture nonlinear decision boundaries and interactions among technical indicators. In financial data, this is valuable because the effect of one indicator is often conditional on another. For example, a momentum signal may behave differently under high volatility than under low volatility.

Unlike ARIMAX, XGBoost does not impose linearity or explicit temporal parametric structure. Instead, it learns a discriminative mapping from engineered features directly to directional probability.

3.3 LSTM

The third base learner is an LSTM implemented in TensorFlow/Keras. The model uses 20-day rolling windows with the following sequential inputs:
trend_signal
vol10
rsi
lag_1 to lag_5

The architecture is:
Input
LSTM with 32 hidden units
Dropout with rate 0.2
Dense sigmoid output

Features are scaled using MinMaxScaler fitted only on the training set. The LSTM is designed to capture temporal dependence that static tabular models may miss. This includes persistence, multi-step interactions, and latent sequential states that may not be easily summarized by one-day lag features alone.

3.4 Stacking and Context-Aware Stacking

Two stackers are implemented.

The first is logistic regression stacking using only base model outputs:
xgb_prob
lstm_prob
arimax_prob

This model serves as a simple static ensemble baseline. It tests whether a linear combination of the base models is more stable than any individual model.

The second is context-aware stacking with an XGBoost meta-model. In the final version of the project, the meta-model uses:
xgb_prob
lstm_prob
arimax_prob
vol10
rsi
trend_signal

This reduced context set was chosen after broader context spaces overfit strongly. Earlier versions also included additional context variables such as vol20, momentum5, MACD, and volume_change. Although these richer context spaces improved in-sample flexibility, they made the meta-learning layer too unstable given the size of the validation window.

The final adaptive stacker includes several anti-overfitting modifications:
shallower trees
fewer boosting rounds
stronger L1 and L2 regularization
reduced row and column subsampling
chronological holdout within the validation period
early stopping
validation-based threshold selection

These changes were introduced because the initial context-aware stacker achieved unrealistically high validation metrics while generalizing poorly to the test period. The goal of the final design was therefore not to maximize validation accuracy, but to produce a more credible and stable meta-model.

Stacking was chosen over other ensemble approaches because the base learners in this project are intentionally heterogeneous. Bagging is most effective when combining many similar high-variance learners trained on resampled versions of the same problem, while boosting is designed to build a stronger learner by sequentially correcting errors within one model family. Neither approach is naturally suited to combining an econometric model, a tree-based classifier, and a recurrent neural network that each encode different assumptions about the data. Stacking is more appropriate because it allows a meta-learner to infer when each base model is reliable and how their signals should be combined. The context-aware version extends this idea further by allowing that combination rule to depend on market regime features rather than remaining fixed across all conditions.

3.5 Evaluation

The project uses both classification and financial evaluation.

Classification metrics:
accuracy
precision
recall
F1 score
confusion matrix

Financial metrics:
cumulative return
Sharpe ratio

The financial strategy is a simple long/flat rule:
long if prediction = 1
flat otherwise

This financial evaluation is not intended to represent a fully deployable trading system. Instead, it provides an additional lens for understanding whether the signals capture economically meaningful directional structure.

Table 1. Summary of model classes, inputs, outputs, intended roles, strengths, and limitations.

Model: ARIMAX
Input: Forward-return series with exogenous variables
Output: 5-day forward return forecast converted to direction
Main role: Linear econometric baseline
Strength: Interpretable and suited for smooth linear structure
Limitation: Weak for complex nonlinear decision boundaries

Model: XGBoost
Input: Full engineered feature set
Output: Probability of positive 5-day direction
Main role: Nonlinear tabular classifier
Strength: Captures nonlinear feature interactions well
Limitation: Does not explicitly model sequence structure

Model: LSTM
Input: 20-day rolling windows of sequential indicators
Output: Probability of positive 5-day direction
Main role: Sequence-based temporal model
Strength: Captures temporal dependence and persistent hidden-state behavior
Limitation: More data-hungry and less interpretable

Model: Logistic Stacking
Input: Base model probabilities
Output: Final ensemble probability
Main role: Static ensemble baseline
Strength: Simple and often stable
Limitation: Cannot adapt weights by regime

Model: Context-Aware Stacking (XGBoost Meta-Model)
Input: Base model probabilities plus vol10, rsi, and trend_signal
Output: Final ensemble probability
Main role: Adaptive regime-aware ensemble
Strength: Can learn conditional model reliability with compact context
Limitation: Still vulnerable to unstable meta-learning

4. Experiments on Synthetic Data

Synthetic experiments were designed to test whether model performance aligns with the structure of the data-generating process. Synthetic returns were generated first and converted into price paths using

P_t = P_(t-1) * (1 + r_t)

Synthetic volume was generated as a noisy function of return magnitude so that the same feature-engineering pipeline used for SPY could be reused without major changes. In this way, the synthetic experiments were not isolated toy classifiers, but controlled versions of the same downstream forecasting pipeline used on real data.

Four synthetic regimes were studied:
1. linear autoregressive
2. nonlinear threshold
3. sequential latent-state
4. regime-switching

The synthetic experiments were designed to test three key properties:
1. linearity versus nonlinearity
2. temporal memory and latent persistence
3. robustness to increasing noise and limited sample size

Noise levels were varied across low, medium, and high settings, and a sample-size sensitivity study was performed for the regime-switching case. Synthetic experiments used 3000 observations per regime by default, split chronologically into 2000 training, 500 validation, and 500 test observations. The smaller sample-size study used 700/200/200, and the medium study used 1200/300/300. This allows the report to evaluate not only which model wins under a given regime, but also how stable that ranking is as the signal-to-noise ratio and available training data change.

4.1 Linear Regime

The linear regime was intended to favor ARIMAX because the underlying return dynamics were approximately linear. The data-generating process was

r_t = 0.25 * r_(t-1) + 0.015 * z_t + error_t

where z_t is a Gaussian exogenous signal and error_t is zero-mean noise whose variance changes across the low, medium, and high noise settings.

The main property tested here is whether a linear econometric model remains best after the task is transformed into a five-day directional classification problem with technical indicators. However, ARIMAX did not dominate the final classification task as cleanly as expected. At medium noise, logistic stacking and context-aware stacking outperformed ARIMAX in test F1. This suggests that even when the underlying return process is linear, the transformation into technical indicators and multi-day directional labels creates a more complex classification surface than the original return equation alone.

4.2 Nonlinear Threshold Regime

The nonlinear regime was intended to favor XGBoost. A representative form of the data-generating process was a threshold rule of the form

r_t = 0.15 * r_(t-1) + 0.05 * 1(z_t > 0) - 0.04 * 1(z_t <= 0) + error_t

where the indicator functions introduce piecewise nonlinear behavior.

The main property tested here is whether a nonlinear tabular learner is more effective when directional behavior depends on threshold-type interactions rather than smooth linear structure. XGBoost was indeed competitive, especially at lower noise, but logistic stacking was often strongest overall. This result indicates that the nonlinear feature learner and the other base models captured complementary information. The ensemble was therefore useful not because the nonlinear structure disappeared, but because different models captured different slices of it.

4.3 Sequential Latent-State Regime

The sequential regime was the clearest success case for LSTM. Here a persistent latent state s_t evolves over time and drives returns together with autoregressive memory. A simplified form is

s_t = s_(t-1) with high probability, otherwise 1 - s_(t-1)

r_t = mu_(s_t) + 0.35 * r_(t-1) + 0.10 * r_(t-2) + error_t

This regime tests whether a sequence model benefits when predictive information depends on latent persistence rather than one-step tabular relationships alone. Because the synthetic process contained persistent hidden states and temporal memory, the sequence model had a direct inductive-bias advantage. At low and medium noise, LSTM strongly outperformed XGBoost and remained among the top-performing models overall. This result provides the cleanest confirmation that recurrent models are valuable when predictive information depends on sequential persistence rather than only on static feature interactions.

4.4 Regime-Switching Regime

The regime-switching process was the most important synthetic experiment because it directly tested the main project motivation: whether adaptive stacking could exploit changes in model reliability. Two regimes were mixed:

Regime A: low-volatility linear dynamics
r_t = 0.30 * r_(t-1) + 0.12 * z_t + error_t

Regime B: higher-volatility nonlinear threshold dynamics
r_t = 0.10 * r_(t-1) + 0.08 * 1(z_t > 0) - 0.06 * 1(z_t <= 0) + error_t

The active regime changes over time according to a persistent switching process.

This experiment tests both regime sensitivity and the effect of limited meta-training data. Context-aware stacking often achieved strong validation performance, but its out-of-sample test behavior was less stable. Logistic stacking was frequently more robust, and LSTM remained strong in many settings. These results foreshadowed the real-data overfitting issue and suggest that adaptive meta-learning is itself highly sensitive to sample size and regime composition.

4.5 Synthetic Summary

The synthetic experiments support four conclusions:
1. LSTM is strongest when temporal memory matters.
2. XGBoost is useful when threshold-type nonlinearity matters.
3. ARIMAX is a meaningful linear baseline but is not always best for final directional classification.
4. Stacking can help in controlled heterogeneous settings, but adaptive stacking is less stable than simpler stacking.

These conclusions map directly onto the tested properties. The linear and nonlinear regimes test how the models respond to smooth structure versus threshold behavior. The sequential regime tests whether temporal memory provides an advantage to recurrent modeling. The regime-switching study, together with the noise and sample-size variations, tests whether adaptive meta-learning remains stable when the apparent best model changes over time and when the effective amount of meta-training data is limited.

These conclusions are supported quantitatively in Table 2, which summarizes the medium-noise test results across regimes.

Table 2. Synthetic experiment summary under medium-noise conditions.

Regime: Linear
XGBoost: test accuracy 0.4465, test F1 0.2171
ARIMAX: test accuracy 0.4202, test F1 0.1433
LSTM: test accuracy 0.4101, test F1 0.0068
Logistic Stacking: test accuracy 0.5596, test F1 0.6746
Context-Aware Stacking: test accuracy 0.5374, test F1 0.6617
Observation: Logistic stacking and context-aware stacking outperform ARIMAX on the final classification task despite the underlying linear return process.

Regime: Nonlinear
XGBoost: test accuracy 0.5071, test F1 0.5643
ARIMAX: test accuracy 0.5374, test F1 0.6336
LSTM: test accuracy 0.4667, test F1 0.4359
Logistic Stacking: test accuracy 0.5838, test F1 0.6720
Context-Aware Stacking: test accuracy 0.4848, test F1 0.2975
Observation: Logistic stacking is strongest overall, while XGBoost is competitive and context-aware stacking remains unstable.

Regime: Sequential
XGBoost: test accuracy 0.4428, test F1 0.0652
ARIMAX: test accuracy 0.5162, test F1 0.6267
LSTM: test accuracy 0.6263, test F1 0.7102
Logistic Stacking: test accuracy 0.6328, test F1 0.6840
Context-Aware Stacking: test accuracy 0.6350, test F1 0.7091
Observation: LSTM is the strongest base learner, confirming that temporal persistence is the key signal in this regime.

Regime: Regime Switching
XGBoost: test accuracy 0.4828, test F1 0.0791
ARIMAX: test accuracy 0.5253, test F1 0.5011
LSTM: test accuracy 0.5354, test F1 0.6526
Logistic Stacking: test accuracy 0.5616, test F1 0.6851
Context-Aware Stacking: test accuracy 0.5051, test F1 0.5386
Observation: Logistic stacking and LSTM are more stable than context-aware stacking when model reliability changes across regimes.

5. Experiments on Real Data

The real-data experiment uses SPY from 2013-01-01 to 2024-01-01, downloaded from yfinance. SPY is the SPDR S&P 500 ETF Trust and serves as a liquid, diversified proxy for large-cap U.S. equities. It was selected because it is smoother and less idiosyncratic than many single-company stocks, making it a reasonable first benchmark for an ensemble forecasting system. Adjusted close is used as the main price series and volume is used for additional features. Missing values are handled during preprocessing, and the final dataset is indexed chronologically to preserve temporal consistency.

Although the report focuses on tables for space efficiency, the implementation also generates diagnostic visual outputs such as feature-importance plots, training curves, confusion matrices, and cumulative-return comparisons. These were used during development to inspect model behavior and confirm that the reported tabular results were not masking obvious pathologies.

[Figure 1 here]
Representative synthetic price paths for the linear, nonlinear, sequential, and regime-switching data-generating processes.

[Figure 2 here]
Held-out SPY cumulative-return comparison for the major models under the long/flat strategy.

The experimental design uses each model as a baseline against the others. ARIMAX is the classical econometric baseline, XGBoost is the nonlinear tabular baseline, LSTM is the sequential deep-learning baseline, logistic stacking is the static ensemble baseline, and context-aware stacking is the adaptive ensemble under study. Base models are trained on the train set. Validation predictions are used to train the stackers. The test set is reserved strictly for final evaluation.

Validation results are summarized first because they show how strong the models appear before final held-out evaluation. These metrics are presented in Table 3.

Table 3. Validation results on SPY.

Columns: Model | Accuracy | Precision | Recall | F1

XGBoost | 0.4741 | 0.4672 | 0.9145 | 0.6185
ARIMAX | 0.5020 | 0.4722 | 0.5812 | 0.5211
LSTM | 0.4661 | 0.4661 | 1.0000 | 0.6359
Always Predict Up | 0.4661 | 0.4661 | 1.0000 | 0.6359
Logistic Stacking | 0.5339 | 0.5000 | 0.0940 | 0.1583
Context-Aware Stacking (XGBoost Meta-Model) | 0.6375 | 0.6512 | 0.4786 | 0.5517

The held-out test comparison is summarized in Table 4.

Table 4. Held-out SPY test results.

Columns: Model | Accuracy | Precision | Recall | F1 | Sharpe Ratio | Cumulative Return

XGBoost | 0.5551 | 0.6146 | 0.8077 | 0.6981 | 3.4872 | 1.5149
ARIMAX | 0.4286 | 0.5889 | 0.3397 | 0.4309 | 2.2593 | 0.5494
LSTM | 0.6367 | 0.6367 | 1.0000 | 0.7781 | 4.1685 | 2.1429
Always Predict Up | 0.6367 | 0.6367 | 1.0000 | 0.7781 | not applicable | not reported
Logistic Stacking | 0.3878 | 0.8750 | 0.0449 | 0.0854 | 2.1861 | 0.2177
Context-Aware Stacking (XGBoost Meta-Model) | 0.4122 | 0.7143 | 0.1282 | 0.2174 | 2.8530 | 0.4657

[Figure 3 here]
Example confusion matrix for the XGBoost model on the SPY test set. This figure is useful because it provides a cleaner visual summary of the strongest nontrivial learned baseline, showing a more balanced tradeoff between true positives and false positives than the adaptive ensemble.

Interpretation:
The real-data baseline comparison reveals an important nuance. On both validation and test data, the LSTM metrics match an always-predict-up baseline exactly, which means that its apparent strength comes from predicting the positive class for every observation rather than learning a discriminative boundary. This makes XGBoost the strongest nontrivial learned model on SPY, while ARIMAX remains a useful interpretable baseline. The context-aware stacker improved after regularization, threshold tuning, and reduction of the context feature set, but it still does not outperform XGBoost on held-out SPY data.

This result is important because it changes the project conclusion from a simplistic “ensemble wins” narrative to a more meaningful research conclusion. The synthetic experiments show that ensemble learning can help when the structure is controlled and heterogeneous. The real-data experiment shows that robust adaptive stacking is harder to train than expected because the meta-learning stage is itself noisy, regime-dependent, and data-limited. It also shows why naive baselines matter: strong headline accuracy can be misleading when class balance shifts over time.

6. Discussion

The strongest aspect of the project is the connection between modeling assumptions and empirical evaluation. The synthetic experiments isolate model inductive bias under known structure, while the SPY experiment tests whether those conclusions remain meaningful in a real financial time series.

The results show that no single model class is universally best. LSTM performs best in the synthetic sequential regime, where temporal persistence is the true source of signal. XGBoost is effective for nonlinear interactions and is the strongest nontrivial learned model on real SPY data. ARIMAX provides an interpretable linear baseline. Stacking can improve performance in controlled heterogeneous settings.

At the same time, the project shows that adaptive stacking is difficult to train robustly on real financial data. The context-aware stacker can fit validation regimes strongly but generalize inconsistently. The simplified final stacker is better than earlier overfit versions, but it remains weaker than the best base learner on SPY.

This is not a failure of the project. It is an important result. The work demonstrates that meta-learning in finance is itself a regime-sensitive learning problem. If the context representation is too flexible or the validation window is too limited, the adaptive ensemble can become more fragile than the single models it is trying to improve upon.

The project also has several limitations. First, the real-data analysis is restricted to SPY, so the conclusions may not transfer directly to more volatile or more idiosyncratic stocks such as TSLA or NVDA. Second, the feature set is limited to technical and volume-based variables; it does not include macroeconomic, fundamental, or sentiment inputs. Third, the backtest ignores transaction costs, slippage, and explicit risk management. Finally, the meta-learning framework would likely benefit from stronger walk-forward or out-of-fold temporal training rather than a single validation-year approach.

A final strength of the project is that it includes explicit analysis of failure modes. Rather than reporting only the strongest-looking validation numbers, the project documents the overfitting behavior of the original adaptive stacker and the subsequent improvements produced by simplification and regularization. This makes the project more credible as an empirical machine learning study.

7. Conclusion

This project investigated five-day stock direction forecasting using ARIMAX, XGBoost, LSTM, logistic stacking, and context-aware stacking.

The synthetic experiments showed that model inductive bias matters. LSTM performed best in sequential regimes, XGBoost was effective in nonlinear settings, ARIMAX remained a useful linear baseline, and stacking often improved performance under controlled heterogeneous structure. However, adaptive stacking was generally less stable than simpler stacking.

The SPY experiments showed that the raw best test metrics were achieved by a trivial always-up pattern, which the LSTM matched exactly. Once that naive baseline is taken into account, XGBoost becomes the strongest nontrivial learned model, with ARIMAX as an interpretable benchmark. The final context-aware stacker improved after regularization and simplification, but it still did not outperform XGBoost on real held-out data.

The final conclusion is therefore nuanced rather than purely optimistic. Ensemble learning can help when model skill varies across controlled regimes, but robust adaptive stacking is difficult to deploy reliably on noisy real financial data. In this project, that difficulty became one of the main findings rather than a failure of the study. As a result, the project demonstrates not only implementation of multiple forecasting models, but also a careful empirical investigation of when adaptive ensemble learning helps and when it fails to generalize.

References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. Time Series Analysis: Forecasting and Control.
2. Chen, T., and Guestrin, C. XGBoost: A Scalable Tree Boosting System.
3. Hochreiter, S., and Schmidhuber, J. Long Short-Term Memory.
4. Wolpert, D. Stacked Generalization.
5. Documentation for statsmodels, xgboost, tensorflow, and yfinance.
