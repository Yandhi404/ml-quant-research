Machine Learning Quantitative Research

A machine learning-based stock selection research project using historical Chinese A-share market data.

This project explores whether a Random Forest model can rank stocks by their subsequent five-trading-day returns, and compares its performance with a traditional momentum strategy and random stock selection.

This is an educational research project, not investment advice. The backtest does not establish that the strategy is profitable in live trading.

1. Research Objective

The main research question is:

Can a machine learning model provide better stock-selection results than a simple momentum strategy or random selection under the same backtesting assumptions?

The project covers the full workflow:

* Historical market data processing
* Feature engineering
* Time-based dataset splitting
* Random Forest model training
* Prediction evaluation
* Strategy comparison
* Portfolio backtesting and accounting reconciliation

2. Dataset

The project uses historical daily stock data for approximately 300 CSI 300 constituent stocks.

* Data period: January 2021 to October 2026
* Training period: Through December 2024
* Validation period: January to December 2025
* Test period: January to October 2026
* Market: Chinese A-share stocks

The dataset contains daily open, high, low, close, volume, and other market fields.

Important limitation: The constituent list represents current constituents rather than a verified historical constituent universe. This introduces potential survivorship bias.

3. Features and Target

The model uses four input features:

Feature	Description
return_1d	One-day historical return
return_5d	Five-day historical return
return_20d	Twenty-day historical return
volatility_20d	Twenty-day rolling volatility

The prediction target is the subsequent five-trading-day return:

target_return_5d = close.shift(-5) / open.shift(-1) - 1

The intended trading timeline is:

1. Generate a signal after the close on day t.
2. Enter at the next trading day’s open.
3. Exit at the close on day t+5.

The target uses future prices only to construct the label. Future target values are not input features.

4. Model

The baseline model is a RandomForestRegressor implemented with scikit-learn.

Parameter	Value
Number of trees	100
Maximum depth	8
Minimum samples per leaf	100
Random seed	42

The model is trained on historical observations and evaluated on later dates to reduce look-ahead bias.

5. Strategy Comparison

Three strategies are evaluated using a shared backtesting framework:

1. Random Forest: Select the top 20% of stocks ranked by predicted five-day return.
2. 20-day Momentum: Select the top 20% ranked by historical 20-day return.
3. Random Selection: Randomly select 20% of the available stocks using a fixed random seed.

Backtesting assumptions

* Initial portfolio capital: CNY 100,000
* Holding period: Five trading days
* Capital allocation: Five rotating sleeves
* Trading cost assumption: 0.15% per side
* Entry: Next trading day’s open
* Exit: Fifth trading day’s close

Fractional shares are used for simplified portfolio accounting. The simulation does not fully model A-share board-lot rules, limit-up and limit-down execution constraints, suspensions, or all corporate actions.

6. Results

Test-period portfolio backtest results:

Strategy	Final Equity	Total Return	Maximum Drawdown	Approx. Annualized Sharpe
Random Forest	CNY 94,241.98	-5.76%	-22.35%	-0.113
20-day Momentum	CNY 92,008.00	-7.99%	-28.39%	-0.317
Random Selection	CNY 86,002.22	-14.00%	-15.75%	-1.239

All three strategies produced negative returns during the test period.

The Random Forest strategy performed better than the two comparison strategies in this particular backtest. However, this result does not establish statistically significant or persistent predictive value.

The portfolio accounting reconciliation difference was approximately zero for all three strategies.

Interpretation

* The model performed relatively better than the chosen baselines in this test period.
* The model did not produce a positive portfolio return after the assumed trading costs.
* Further statistical testing and more realistic execution assumptions are required before drawing conclusions about practical profitability.

7. Project Structure

ml-quant-research/
├── data/                                      # Historical stock data
├── build_features.py                          # Feature engineering
├── prepare_dataset.py                         # Time-based dataset preparation
├── train_model.py                             # Random Forest training
├── evaluate_signal.py                         # Prediction-based signal evaluation
├── evaluate_momentum.py                       # Momentum baseline evaluation
├── test_signal_significance.py                # Signal significance analysis
├── compare_strategies.py                       # Signal-level strategy comparison
├── test_ml_vs_momentum.py                     # ML vs momentum statistical test
├── backtest_v4.py                             # Single-strategy portfolio backtest
├── compare_backtests.py                       # Unified portfolio comparison
├── test_predictions.csv                       # Out-of-sample predictions
├── strategy_comparison_backtests_summary.csv  # Final comparison metrics
└── strategy_comparison_backtests.png          # Portfolio equity curves

Some generated datasets and output files may be omitted from version control depending on their size.

8. Limitations and Future Work

Known limitations include:

* Potential survivorship bias from using current index constituents.
* Simplified trading-cost assumptions.
* Incomplete simulation of A-share trading restrictions and execution failures.
* Use of adjusted historical prices, which may not reproduce actual cash and share accounting exactly.
* Overlapping five-day return labels, which complicate statistical inference.
* No demonstrated statistically significant or persistent excess return.

Potential next steps include testing on a longer out-of-sample period, using point-in-time historical constituents, improving execution simulation, and applying statistical tests that account for overlapping returns.

9. Reproducibility

The model and random-selection baseline use fixed random seeds where applicable.

The exact results may depend on the source data, preprocessing, package versions, and the implementation of the backtesting engine.

This project is intended to demonstrate an end-to-end quantitative research workflow rather than a production-ready trading system.

Disclaimer

This repository is for educational and research purposes only. It does not constitute investment advice or a recommendation to buy or sell any security.