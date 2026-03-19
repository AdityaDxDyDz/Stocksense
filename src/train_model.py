import os
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter


FEATURES = [
    "Return",
    "MA_5",
    "MA_10",
    "MA_20",
    "Volatility_10",
    "Momentum_10",
    "RSI",
    "MACD",
    "Price_Change",
    "Trend",
    "Sentiment",
    "Market_Regime",
    "Volume_Change"
]


def train_model():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "featured_data.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\n❌ featured_data.csv NOT FOUND\nExpected at: {DATA_PATH}\n👉 Run feature_eng.py first"
        )

    print("📂 Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset Shape:", df.shape)

    # Clean dataset
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES + ["Target"], inplace=True)

    # Sort properly
    df.sort_values(["Ticker", "Date"], inplace=True)

    # Optional: sample if dataset too large
    if len(df) > 2_000_000:
        print("⚠ Large dataset detected, sampling for faster training")
        df = df.sample(2_000_000, random_state=42)

    # Train data
    X = df[FEATURES]
    y = df["Target"]

    print("Feature matrix:", X.shape)

    # Time-aware split (no shuffle)
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Handle class imbalance
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / max(counter[1], 1)

    print("Class distribution:", counter)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist"   # faster for big datasets
    )

    print("🚀 Training XGBoost model...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Model Accuracy: {accuracy:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES
        },
        MODEL_PATH
    )

    print(f"\n✅ Model saved at:\n{MODEL_PATH}")


if __name__ == "__main__":
    train_model()