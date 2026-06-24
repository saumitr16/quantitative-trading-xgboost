#Quantitative Trading Strategy using XGBoost

## Overview

This project implements an end-to-end machine learning pipeline for quantitative stock selection and return prediction using historical market data from NIFTY 50 stocks.

The system collects historical stock data, generates technical and market-based features, trains machine learning models to forecast future returns, and evaluates trading performance through stock ranking and portfolio backtesting.

---

## Features

* Multi-stock dataset covering 21 NIFTY stocks
* 50,000+ historical observations
* Technical indicator feature engineering
* Market-relative feature generation using NIFTY index
* Return prediction using XGBoost
* Classification and regression-based forecasting
* Time-series aware train-test split
* Feature importance analysis
* Portfolio ranking framework
* Backtesting and performance evaluation

---

## Dataset

### Stocks

Example stocks used:

* RELIANCE
* TCS
* INFY
* HDFCBANK
* ICICIBANK
* SBIN
* ITC
* LT
* BHARTIARTL
* ASIANPAINT
* and other NIFTY constituents

### Time Period

* January 2015 – December 2024

### Data Source

* Yahoo Finance (via yfinance)

---

## Project Structure

```text
quant_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── labeling.py
│   ├── prepare_dataset.py
│   ├── build_market_features.py
│   ├── train_master.py
│   ├── train_classifier.py
│   └── backtest.py
|──check.py(just for intermediate validations and checks)
├── requirements.txt
└── README.md
```

---

## Feature Engineering

### Technical Indicators

* SMA 10
* SMA 20
* SMA 50
* RSI (14)
* MACD
* MACD Signal
* Bollinger Upper Band
* Bollinger Lower Band

### Momentum Features

* 1-Day Return
* 5-Day Return

### Volatility Features

* 5-Day Volatility
* 20-Day Volatility

### Trend Features

* Close / SMA20 Ratio
* Close / SMA50 Ratio
* SMA10 / SMA50 Ratio

### Market Features

* NIFTY 1-Day Return
* NIFTY 5-Day Return

---

## Prediction Target

Future 5-day average return:

```python
future_avg = (
    close.shift(-5)
    .rolling(5)
    .mean()
)

target_return = (
    future_avg / close - 1
)
```

Classification target:

```python
target_class = (
    target_return > 0.02
).astype(int)
```

---

## Model

### Regression

* XGBoost Regressor

Objective:

Predict future stock returns.

### Classification

* XGBoost Classifier

Objective:

Predict whether future return exceeds 2%.

---

## Evaluation Methodology

### Train/Test Split

Time-series aware split:

```text
2015 ----------- 2022 | 2023 -------- 2024
       TRAIN         |      TEST
```

No future information is used during training.

---

## Results

### Regression Performance

| Metric               | Value  |
| -------------------- | ------ |
| MAE                  | 0.0147 |
| RMSE                 | 0.0196 |
| Directional Accuracy | 52.8%  |

### Classification Performance

| Metric   | Value  |
| -------- | ------ |
| ROC-AUC  | 0.55   |
| Accuracy | 84.7%* |

*Accuracy is inflated due to class imbalance and should not be used as the primary metric.

### Ranking Evaluation (Out-of-Sample)

| Metric                | Value  |
| --------------------- | ------ |
| Market Average Return | 0.216% |
| Top-5 Average Return  | 0.258% |
| Top-5 Win Rate        | 61.8%  |

---

## Backtesting

A ranking-based portfolio strategy was evaluated:

1. Predict future returns for all stocks.
2. Rank stocks by predicted return.
3. Select Top-5 stocks.
4. Allocate capital equally.
5. Rebalance periodically.

### Backtest Metrics

| Metric       | Value  |
| ------------ | ------ |
| CAGR         | 86.1%  |
| Sharpe Ratio | 3.28   |
| Max Drawdown | -22.9% |
| Win Rate     | 61.8%  |

---

## Technologies Used

* Python
* Pandas
* NumPy
* XGBoost
* Scikit-Learn
* Pandas TA
* Yahoo Finance API
* Matplotlib

---

## Key Learnings

* Time-series machine learning pipelines
* Financial feature engineering
* Quantitative stock ranking
* Market-relative signal generation
* Portfolio construction
* Backtesting methodologies
* Model evaluation on unseen financial data

---

## Future Improvements

* Walk-forward validation
* Sector-aware features
* Fundamental data integration
* News sentiment analysis
* Transaction cost modeling
* Risk-adjusted portfolio optimization
* Transformer-based time-series models

---

## Disclaimer

This project is intended for educational and research purposes only. It should not be considered financial advice or a production trading system.
