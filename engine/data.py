"""
Fetch and cache OHLCV data from Yahoo Finance.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import hashlib, json
from datetime import datetime

CACHE_DIR = Path(".cache/stock_data")


def _cache_key(ticker: str, start: str, end: str, interval: str) -> str:
    return hashlib.md5(f"{ticker}-{start}-{end}-{interval}".encode()).hexdigest()


def fetch(ticker: str, start: str = "2020-01-01", end: str | None = None, interval: str = "1d", use_cache: bool = True) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    key = _cache_key(ticker, start, end, interval)
    cache_path = CACHE_DIR / f"{key}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

    return df


def multi_fetch(tickers: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    return {t: fetch(t, **kwargs) for t in tickers}
