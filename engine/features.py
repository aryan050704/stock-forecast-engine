"""
Technical indicator feature engineering for OHLCV data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    bandwidth = (upper - lower) / (mid + 1e-9)
    return pd.DataFrame({"bb_upper": upper, "bb_mid": mid, "bb_lower": lower, "bb_pct_b": pct_b, "bb_width": bandwidth})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff())
    return (direction * df["Volume"]).cumsum()


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["sma_10"] = sma(close, 10)
    df["sma_20"] = sma(close, 20)
    df["sma_50"] = sma(close, 50)
    df["ema_12"] = ema(close, 12)
    df["ema_26"] = ema(close, 26)
    df["rsi_14"] = rsi(close, 14)
    df["atr_14"] = atr(df, 14)
    df["obv"] = obv(df)

    macd_df = macd(close)
    df = pd.concat([df, macd_df], axis=1)

    bb_df = bollinger_bands(close)
    df = pd.concat([df, bb_df], axis=1)

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = close.shift(lag)
        df[f"return_lag_{lag}"] = close.pct_change(lag)

    # Price ratios
    df["price_to_sma20"] = close / df["sma_20"]
    df["high_low_ratio"] = df["High"] / (df["Low"] + 1e-9)
    df["close_open_ratio"] = close / df["Open"]

    # Target: next-day return direction (1=up, 0=down)
    df["target"] = (close.shift(-1) > close).astype(int)
    df["future_return"] = close.shift(-1) / close - 1

    return df.dropna()
