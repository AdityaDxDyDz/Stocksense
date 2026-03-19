import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split


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


def evaluate_model():

    os.makedirs("plots", exist_ok=True)

    model = joblib.load("models/xgboost_model.pkl")

    df = pd.read_csv("data/processed/featured_data.csv")

    X = df[FEATURES]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    y_pred = model.predict(X_test)

    print("\n📊 MODEL PERFORMANCE")
    print("---------------------")

    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title("Confusion Matrix")

    plt.savefig("plots/confusion_matrix.png")

    plt.close()

    importance = model.feature_importances_

    plt.figure(figsize=(8,6))

    plt.barh(FEATURES, importance)

    plt.title("Feature Importance (XGBoost)")

    plt.xlabel("Importance Score")

    plt.tight_layout()

    plt.savefig("plots/feature_importance.png")

    plt.close()

    print("\n✅ Evaluation plots saved in /plots folder")


if __name__ == "__main__":

    evaluate_model()