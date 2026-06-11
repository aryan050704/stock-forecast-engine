# Stock Forecast Engine

Multi-model stock price **direction forecasting** system with technical indicator feature engineering, XGBoost + LSTM ensemble, and a backtesting framework.

## Architecture

```
yfinance → OHLCV Data → Feature Engineering (20+ indicators)
                               ↓
              ┌────────────────┴─────────────────┐
           XGBoost                              LSTM
              └────────────────┬─────────────────┘
                          Ensemble (60/40)
                               ↓
                     Backtest vs Buy & Hold
```

## Features

### Feature Engineering (engine/features.py)
- **Trend**: SMA(10/20/50), EMA(12/26)
- **Momentum**: RSI(14), MACD + Signal + Histogram
- **Volatility**: Bollinger Bands (upper/mid/lower/width/%B), ATR(14)
- **Volume**: OBV (On-Balance Volume)
- **Lag features**: Close price and returns at lag 1/2/3/5/10

### Models (engine/models.py)
- **XGBoost**: gradient boosted trees with time-series train/test split
- **LSTM**: 2-layer LSTM with dropout, trained with sliding window sequences
- **Ensemble**: weighted average of XGBoost + LSTM probabilities

### Backtesting (engine/backtest.py)
- Signal-based strategy with transaction cost simulation
- Metrics: total return, annualized return, Sharpe ratio, max drawdown, win rate
- Comparison against Buy & Hold benchmark

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer
This is a research/portfolio project. Do not use for actual trading decisions.

## Tech Stack
`Python` `XGBoost` `PyTorch` `yfinance` `Scikit-learn` `Streamlit` `Pandas` `NumPy` `Matplotlib`
