from feature_engineering import prepare_features
from train_model import train_model
from evaluate import evaluate_model
from predict import predict_next_7_days


def main():

    print("=" * 55)
    print("🚀 STOCKSENSE : STOCK PRICE MOVEMENT PREDICTION SYSTEM")
    print("=" * 55)

    print("\n[1] Feature Engineering Started")
    prepare_features()

    print("\n[2] Training XGBoost Model")
    train_model()

    print("\n[3] Evaluating Model")
    evaluate_model()

    symbol = input("\nEnter Stock Symbol (example: AAPL): ").upper()

    print("\n[4] Predicting Next 7 Days")

    result = predict_next_7_days(symbol)

    print("\nPrediction Result:")
    print(result)

    print("\n✅ Pipeline executed successfully!")


if __name__ == "__main__":
    main()