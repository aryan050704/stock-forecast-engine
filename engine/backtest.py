"""
Simple backtesting framework for signal-based strategies.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestResult:
    strategy: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    equity_curve: pd.Series


def _sharpe(returns: pd.Series, risk_free: float = 0.02) -> float:
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def run_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
    strategy_name: str = "ML Signal",
) -> BacktestResult:
    prices = df["Close"].values[-len(signals):]
    n = len(prices)

    portfolio = [initial_capital]
    position = 0.0
    trades = 0
    daily_returns = []

    for i in range(1, n):
        signal = signals[i - 1]
        price_change = (prices[i] - prices[i - 1]) / prices[i - 1]

        if signal == 1 and position == 0:
            position = 1.0
            portfolio.append(portfolio[-1] * (1 - transaction_cost))
            trades += 1
        elif signal == 0 and position == 1:
            position = 0.0
            portfolio.append(portfolio[-1] * (1 - transaction_cost))
            trades += 1
        else:
            portfolio.append(portfolio[-1])

        if position == 1:
            portfolio[-1] *= (1 + price_change)

        daily_returns.append(portfolio[-1] / portfolio[-2] - 1 if len(portfolio) > 1 else 0)

    equity = pd.Series(portfolio, index=df.index[-len(portfolio):])
    returns = pd.Series(daily_returns)

    total_return = (portfolio[-1] - initial_capital) / initial_capital
    n_years = n / 252
    annualized = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

    return BacktestResult(
        strategy=strategy_name,
        total_return=round(total_return * 100, 2),
        annualized_return=round(annualized * 100, 2),
        sharpe_ratio=round(_sharpe(returns), 3),
        max_drawdown=round(_max_drawdown(equity) * 100, 2),
        win_rate=round(win_rate * 100, 2),
        n_trades=trades,
        equity_curve=equity,
    )


def buy_and_hold(df: pd.DataFrame, initial_capital: float = 10_000.0) -> BacktestResult:
    prices = df["Close"]
    equity = initial_capital * prices / prices.iloc[0]
    returns = prices.pct_change().dropna()
    total_return = (equity.iloc[-1] - initial_capital) / initial_capital
    n_years = len(prices) / 252
    annualized = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    return BacktestResult(
        strategy="Buy & Hold",
        total_return=round(total_return * 100, 2),
        annualized_return=round(annualized * 100, 2),
        sharpe_ratio=round(_sharpe(returns), 3),
        max_drawdown=round(_max_drawdown(equity) * 100, 2),
        win_rate=round(float((returns > 0).mean()) * 100, 2),
        n_trades=1,
        equity_curve=equity,
    )
