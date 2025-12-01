from src.load_data import load_covid_data
from src.preprocess import preprocess
from src.model_lstm import build_lstm
from src.train import train_model
from src.predict_future import predict_future

def main():
    # Load data
    df = load_covid_data()

    # Change country here (India, Afghanistan, US, etc.)
    X, y, scaler = preprocess(df, country="India")

    # Build + train model
    model = build_lstm((X.shape[1], X.shape[2]))
    train_model(model, X, y)

    # Predict next 10 weeks
    future = predict_future(model, X[-1], scaler, steps=10)
    print("\nPredicted COVID Cases (Next 10 Weeks):")
    print(future)

if __name__ == "__main__":
    main()

