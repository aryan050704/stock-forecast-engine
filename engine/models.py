"""
LSTM + XGBoost ensemble for stock price direction forecasting.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import dataclass


@dataclass
class ForecastResult:
    model: str
    accuracy: float
    f1: float
    predictions: np.ndarray
    probabilities: np.ndarray
    feature_importance: pd.Series | None = None


FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14", "atr_14",
    "macd", "macd_signal", "macd_hist", "bb_pct_b", "bb_width",
    "close_lag_1", "close_lag_2", "close_lag_3", "return_lag_1",
    "return_lag_2", "return_lag_3", "price_to_sma20", "high_low_ratio", "close_open_ratio"
]


def _get_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].values
    y = df["target"].values
    return X, y


def train_xgboost(df: pd.DataFrame) -> ForecastResult:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("Install xgboost: pip install xgboost")

    X, y = _get_xy(df)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                          random_state=42, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    cols = [c for c in FEATURE_COLS if c in df.columns]
    fi = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)

    return ForecastResult(
        model="XGBoost",
        accuracy=round(accuracy_score(y_test, preds), 4),
        f1=round(f1_score(y_test, preds, zero_division=0), 4),
        predictions=preds,
        probabilities=probs,
        feature_importance=fi,
    )


def train_lstm(df: pd.DataFrame, lookback: int = 20) -> ForecastResult:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("Install PyTorch: pip install torch")

    X, y = _get_xy(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Build sequences
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    Xs, ys = np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    split = int(len(Xs) * 0.8)
    X_train, X_test = torch.tensor(Xs[:split]), torch.tensor(Xs[split:])
    y_train, y_test = torch.tensor(ys[:split]).unsqueeze(1), torch.tensor(ys[split:]).unsqueeze(1)

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden=64, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid())

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMModel(Xs.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = model(X_test).squeeze().numpy()

    preds = (probs > 0.5).astype(int)
    y_np = ys[split:]

    return ForecastResult(
        model="LSTM",
        accuracy=round(accuracy_score(y_np, preds), 4),
        f1=round(f1_score(y_np, preds, zero_division=0), 4),
        predictions=preds,
        probabilities=probs,
    )


def ensemble(xgb: ForecastResult, lstm: ForecastResult, weight_xgb: float = 0.6) -> ForecastResult:
    min_len = min(len(xgb.probabilities), len(lstm.probabilities))
    xgb_prob = xgb.probabilities[-min_len:]
    lstm_prob = lstm.probabilities[-min_len:]
    combined = weight_xgb * xgb_prob + (1 - weight_xgb) * lstm_prob
    preds = (combined > 0.5).astype(int)
    y_test = xgb.predictions[-min_len:]  # use same test set

    return ForecastResult(
        model="Ensemble (XGB+LSTM)",
        accuracy=round(float(np.mean(preds == y_test)), 4),
        f1=round(f1_score(y_test, preds, zero_division=0), 4),
        predictions=preds,
        probabilities=combined,
    )
