import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_rsi(series, window=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def load_all_stock_data():

    folder = "stock_data"

    all_data = []

    for file in os.listdir(folder):

        if file.endswith(".csv"):

            path = os.path.join(folder, file)

            ticker = file.replace(".csv", "")

            df = pd.read_csv(path)

            df.reset_index(inplace=True)

            df["Ticker"] = ticker

            df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

            all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)

    return df


def create_features(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)

    df.sort_values(["Ticker", "Date"], inplace=True)

    df["Return"] = df.groupby("Ticker")["Close"].pct_change()

    df["MA_5"] = df.groupby("Ticker")["Close"].rolling(5).mean().reset_index(0, drop=True)
    df["MA_10"] = df.groupby("Ticker")["Close"].rolling(10).mean().reset_index(0, drop=True)
    df["MA_20"] = df.groupby("Ticker")["Close"].rolling(20).mean().reset_index(0, drop=True)

    df["Volume_Change"] = df.groupby("Ticker")["Volume"].pct_change()

    df["Volatility_10"] = (
        df.groupby("Ticker")["Return"]
        .rolling(10)
        .std()
        .reset_index(0, drop=True)
    )

    df["Momentum_10"] = df.groupby("Ticker")["Close"].diff(10)

    df["RSI"] = df.groupby("Ticker")["Close"].transform(compute_rsi)

    ema12 = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=12).mean())
    ema26 = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=26).mean())

    df["MACD"] = ema12 - ema26

    df["Price_Change"] = df.groupby("Ticker")["Close"].diff()

    df["Trend"] = df["MA_5"] - df["MA_20"]

    df["Sentiment"] = 0

    df["Target"] = (
        df.groupby("Ticker")["Close"].shift(-1) > df["Close"]
    ).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)

    cluster_features = df[
        [
            "Return",
            "Volatility_10",
            "Momentum_10",
            "RSI"
        ]
    ]

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(cluster_features)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["Market_Regime"] = kmeans.fit_predict(scaled_features)

    print("✅ Feature engineering completed")
    print("✅ Market regime clustering added")

    return df


def prepare_features():

    df = load_all_stock_data()

    df = create_features(df)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv("data/processed/featured_data.csv", index=False)

    print("✅ Featured data saved")


if __name__ == "__main__":

    prepare_features()