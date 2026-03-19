import matplotlib
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.fetch_news import fetch_news

matplotlib.use("Agg")

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

def compute_rsi(series, window=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))

def get_news_sentiment(news_list):

    analyzer = SentimentIntensityAnalyzer()

    scores = []

    for headline in news_list:

        score = analyzer.polarity_scores(headline)["compound"]

        scores.append(score)

    if len(scores) == 0:
        return 0

    return sum(scores) / len(scores)

def plot_market_regime(df, symbol):

    colors = {0: "green", 1: "red", 2: "blue"}

    labels = {
        0: "Bull Market",
        1: "Bear Market",
        2: "Sideways Market"
    }

    plt.figure(figsize=(12,5))

    plt.plot(df["Date"], df["Close"], color="black", linewidth=1.5, label="Price")

    for regime in df["Market_Regime"].unique():

        subset = df[df["Market_Regime"] == regime]

        plt.scatter(
            subset["Date"],
            subset["Close"],
            color=colors.get(regime, "gray"),
            label=labels.get(regime, f"Regime {regime}"),
            s=15
        )

    plt.title(f"{symbol} Market Regime Detection")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = f"static/regime_{symbol}.png"

    plt.savefig(path)
    plt.close()

    return path

model = joblib.load("models/xgboost_model.pkl")

def fetch_stock(symbol):

    symbol = symbol.upper()

    if symbol.endswith("_NS"):
        symbol = symbol.replace("_NS", ".NS")

    df = yf.download(
        symbol,
        period="5y",
        interval="1d",
        progress=False
    )

    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)

    if df.empty:
        raise ValueError("No data found for symbol: " + symbol) 

    return df

def predict_next_7_days(symbol):

    symbol = symbol.upper()

    df = fetch_stock(symbol)

    if df.empty:
        raise ValueError("Stock data not available")

    df["Return"] = df["Close"].pct_change()
    df["Volume_Change"] = df["Volume"].pct_change().fillna(0.0)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    df["Volatility_10"] = df["Return"].rolling(10).std()

    df["Momentum_10"] = df["Close"].diff(10)

    df["RSI"] = compute_rsi(df["Close"])

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26

    df["Price_Change"] = df["Close"].diff()

    df["Trend"] = df["MA_5"] - df["MA_20"]

    df["Market_Regime"] = np.select(
        [
            df["Trend"] > 0,
            df["Trend"] < 0
        ],
        [0, 1],
        default=2
    )

    clean_symbol = symbol.replace("_NS", "")
    news = fetch_news(clean_symbol + " stock market")

    sentiment_value = get_news_sentiment(news)

    df["Sentiment"] = sentiment_value

    df.dropna(inplace=True)

    history_prices = df["Close"].tail(30).tolist()
    history_dates = df["Date"].dt.strftime("%Y-%m-%d").tail(30).tolist()

    current_regime = df["Market_Regime"].iloc[-1]

    regime_map = {
        0: "Bull Market",
        1: "Bear Market",
        2: "Sideways Market"
    }

    regime_label = regime_map.get(current_regime, "Unknown")

    history = df.tail(50).copy()

    predictions = []
    probabilities = []
    labels = []

    last_close = history["Close"].iloc[-1]

    forecast_prices = [last_close]

    for i in range(7):

        X = history[FEATURES].iloc[-1:].values
        
        prob_up = model.predict_proba(X)[0][1]

        pred = 1 if prob_up >= 0.52 else 0

        predictions.append("UP" if pred == 1 else "DOWN")
        probabilities.append(round(prob_up * 100, 2))

        last_close = forecast_prices[-1]

        volatility = history["Volatility_10"].iloc[-1]
        change = last_close * volatility * (1 if pred == 1 else -1)

        new_close = last_close + change

        forecast_prices.append(round(new_close, 2))

        temp = pd.concat([history["Close"], pd.Series([new_close])])

        new_row = history.iloc[-1].copy()

        new_row["Close"] = new_close
        new_row["Return"] = temp.pct_change().iloc[-1]

        new_row["MA_5"] = temp.rolling(5).mean().iloc[-1]
        new_row["MA_10"] = temp.rolling(10).mean().iloc[-1]
        new_row["MA_20"] = temp.rolling(20).mean().iloc[-1]

        new_row["Volatility_10"] = temp.pct_change().rolling(10).std().iloc[-1]

        new_row["Momentum_10"] = temp.diff(10).iloc[-1]

        new_row["RSI"] = compute_rsi(temp).iloc[-1]

        ema12 = temp.ewm(span=12).mean().iloc[-1]
        ema26 = temp.ewm(span=26).mean().iloc[-1]

        new_row["MACD"] = ema12 - ema26

        new_row["Price_Change"] = new_close - history["Close"].iloc[-1]

        new_row["Trend"] = new_row["MA_5"] - new_row["MA_20"]

        new_row["Sentiment"] = sentiment_value

        new_row["Market_Regime"] = 0 if new_row["Trend"] > 0 else 1 if new_row["Trend"] < 0 else 2

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        labels.append(f"Day {i+1}")

    prices = forecast_prices[1:]

    regime_chart = plot_market_regime(df, symbol)

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "labels": labels,
        "prices": prices,
        "regime": regime_label,
        "sentiment": sentiment_value,
        "news": news,
        "history_dates": history_dates,
        "history_prices": history_prices,
        "regime_chart": regime_chart
    }
    