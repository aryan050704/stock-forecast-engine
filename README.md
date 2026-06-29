# Stock Forecast Engine

A project to predict stock price direction using technical indicators + an XGBoost/LSTM ensemble, with a backtest to check if it actually beats just buy-and-hold.

## Pipeline

```
yfinance OHLCV data -> feature engineering (20+ indicators)
                              |
                  XGBoost  +  LSTM
                              |
                  ensemble (weighted average)
                              |
                  backtest vs buy & hold
```

## Features

### Feature engineering (`engine/features.py`)
- Trend: SMA(10/20/50), EMA(12/26)
- Momentum: RSI(14), MACD + signal + histogram
- Volatility: Bollinger Bands (upper/mid/lower/width/%B), ATR(14)
- Volume: OBV
- Lag features on close price and returns (lag 1/2/3/5/10)

### Models (`engine/models.py`)
- XGBoost with a time-series train/test split (no shuffling, to avoid leakage)
- 2-layer LSTM with dropout, trained on sliding windows
- Ensemble: weighted average of XGBoost + LSTM probabilities (60/40)

### Backtesting (`engine/backtest.py`)
- Signal-based strategy with basic transaction cost simulation
- Metrics: total return, annualized return, Sharpe ratio, max drawdown, win rate
- Compared against a buy & hold baseline on the same period

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack
Python, XGBoost, PyTorch, yfinance, Scikit-learn, Streamlit, Pandas, NumPy, Matplotlib

## Notes / things I'd improve
- LSTM window size is fixed, didn't get to properly tune it against different horizons
- 60/40 ensemble weighting was picked from a few manual trials, not a proper grid search
- Backtest doesn't account for slippage, only flat transaction cost

## Disclaimer
This is a learning/portfolio project, not investment advice. Don't trade on it.
