from src.load_data import load_covid_data
from src.preprocess import preprocess
from src.model_lstm import build_lstm
from src.train import train_model
from src.predict_future import predict_future
from src.visualize import plot_results

def main():
    df = load_covid_data()
    country = "India"   # Change to Afghanistan, US, etc.

    X, y, scaler = preprocess(df, country)

    model = build_lstm((X.shape[1], X.shape[2]))
    train_model(model, X, y)

    future = predict_future(model, X[-1], scaler, steps=10)
    print("\nPredicted COVID Cases (Next 10 Weeks):")
    print(future)

    # Show the graph
    plot_results(df, country, future, scaler)

if __name__ == "__main__":
    main()


