from src.load_data import load_covid_data
from src.preprocess import preprocess
from src.model_lstm import build_lstm
from src.train import train_model
from src.predict_future import predict_future
from src.visualize import plot_results
from src.metrics import calculate_metrics
from src.plot_loss import plot_training_history


def main():
    # Load data
    df = load_covid_data()

    # Select country
    country = "India"    # change if needed

    # Preprocess data
    X, y, scaler = preprocess(df, country)

    # Build model
    model = build_lstm((X.shape[1], X.shape[2]))

    # Train model & get training history
    history = train_model(model, X, y)

    # Plot training vs validation loss
    plot_training_history(history)

    # Predict next 10 weeks
    future = predict_future(model, X[-1], scaler, steps=10)

    print("\nPredicted COVID Cases (Next 10 Weeks):")
    print(future)

    # Evaluation metrics (using last 10 actual weeks)
    last_10_actual = df[df["Entity"] == country]["Weekly cases"].values[-10:]
    rmse, mae, mape = calculate_metrics(last_10_actual, future.flatten())

    print("\n--- Model Evaluation Metrics ---")
    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"MAPE: {mape}%")

    # Plot prediction graph
    plot_results(df, country, future, scaler)


if __name__ == "__main__":
    main()

