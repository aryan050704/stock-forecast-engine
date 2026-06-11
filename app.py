import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from engine.data import fetch
from engine.features import add_all_features
from engine.models import train_xgboost, train_lstm, ensemble
from engine.backtest import run_backtest, buy_and_hold

st.set_page_config(page_title="Stock Forecast Engine", layout="wide")
st.title("Stock Forecast Engine")
st.markdown("Multi-model stock direction forecasting: **XGBoost + LSTM ensemble** with technical indicators and backtesting.")

# --- Sidebar ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker symbol", "AAPL").upper()
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
use_lstm = st.sidebar.checkbox("Include LSTM model (slower)", value=False)
run_btn = st.sidebar.button("Run Forecast", type="primary")

st.sidebar.divider()
st.sidebar.markdown("**Popular tickers:** AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, META, RELIANCE.NS, TCS.NS")

if run_btn:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            raw_df = fetch(ticker, start=str(start_date))
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    st.subheader(f"{ticker} — Price Chart")
    fig0, ax0 = plt.subplots(figsize=(12, 3))
    ax0.plot(raw_df.index, raw_df["Close"], color="#2c3e50", linewidth=1)
    ax0.fill_between(raw_df.index, raw_df["Close"], alpha=0.1, color="#3498db")
    ax0.set_ylabel("Price (USD)")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax0.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig0)

    with st.spinner("Engineering features..."):
        df = add_all_features(raw_df)

    st.metric("Data points after feature engineering", len(df))

    # --- Technical indicators plot ---
    st.subheader("Technical Indicators")
    fig1, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(df.index, df["Close"], label="Close", color="#2c3e50")
    axes[0].plot(df.index, df["sma_20"], label="SMA20", alpha=0.7, color="#3498db")
    axes[0].plot(df.index, df["bb_upper"], linestyle="--", alpha=0.5, color="#e74c3c", label="BB Upper")
    axes[0].plot(df.index, df["bb_lower"], linestyle="--", alpha=0.5, color="#e74c3c", label="BB Lower")
    axes[0].fill_between(df.index, df["bb_upper"], df["bb_lower"], alpha=0.05, color="#e74c3c")
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df["rsi_14"], color="#9b59b6")
    axes[1].axhline(70, linestyle="--", color="red", alpha=0.5)
    axes[1].axhline(30, linestyle="--", color="green", alpha=0.5)
    axes[1].set_ylabel("RSI (14)")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(df.index, df["macd_hist"], color=np.where(df["macd_hist"] >= 0, "#2ecc71", "#e74c3c"), alpha=0.7)
    axes[2].plot(df.index, df["macd"], color="#3498db", label="MACD", linewidth=0.8)
    axes[2].plot(df.index, df["macd_signal"], color="#e74c3c", label="Signal", linewidth=0.8)
    axes[2].set_ylabel("MACD")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

    # --- Models ---
    results = {}
    with st.spinner("Training XGBoost..."):
        try:
            xgb_result = train_xgboost(df)
            results["XGBoost"] = xgb_result
        except Exception as e:
            st.warning(f"XGBoost failed: {e}")
            xgb_result = None

    lstm_result = None
    if use_lstm and xgb_result:
        with st.spinner("Training LSTM (this takes ~1 min)..."):
            try:
                lstm_result = train_lstm(df)
                results["LSTM"] = lstm_result
            except Exception as e:
                st.warning(f"LSTM failed: {e}")

    if xgb_result and lstm_result:
        ens = ensemble(xgb_result, lstm_result)
        results["Ensemble"] = ens

    # --- Model comparison ---
    st.subheader("Model Performance")
    comp = pd.DataFrame([{"Model": r.model, "Accuracy": r.accuracy, "F1 Score": r.f1} for r in results.values()])
    st.dataframe(comp.style.highlight_max(subset=["Accuracy", "F1 Score"], color="#d4edda"), use_container_width=True)

    if xgb_result and xgb_result.feature_importance is not None:
        st.subheader("Feature Importance (XGBoost)")
        fi = xgb_result.feature_importance.head(12)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fi.sort_values().plot(kind="barh", ax=ax2, color="#3498db")
        ax2.set_xlabel("Importance")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    # --- Backtest ---
    st.subheader("Backtest Results")
    bh = buy_and_hold(df)
    backtest_results = [bh]

    best_result = xgb_result or (list(results.values())[-1] if results else None)
    if best_result:
        ml_bt = run_backtest(df, best_result.predictions, strategy_name=f"{best_result.model} Strategy")
        backtest_results.append(ml_bt)

    bt_df = pd.DataFrame([{
        "Strategy": b.strategy, "Total Return (%)": b.total_return,
        "Annual Return (%)": b.annualized_return, "Sharpe": b.sharpe_ratio,
        "Max Drawdown (%)": b.max_drawdown, "Win Rate (%)": b.win_rate,
        "Trades": b.n_trades
    } for b in backtest_results])
    st.dataframe(bt_df, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    for bt in backtest_results:
        ax3.plot(bt.equity_curve.index, bt.equity_curve.values, label=bt.strategy, linewidth=1.5)
    ax3.set_ylabel("Portfolio Value ($)")
    ax3.set_title("Equity Curves")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)

else:
    st.info("Enter a ticker symbol and click **Run Forecast** to start.")
