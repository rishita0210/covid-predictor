from src.load_data import load_covid_data
from src.preprocess import preprocess
from src.train import train_model, get_model
from src.predict_future import predict_future
from src.visualize import plot_results
from src.plot_loss import plot_loss


def main():
    # ---------------------------
    # Load dataset
    # ---------------------------
    df = load_covid_data()

    # Select country
    country = "India"   # Change to "Afghanistan", "United States", etc.

    # ---------------------------
    # Preprocess
    # ---------------------------
    X, y, scaler = preprocess(df, country)

    # ---------------------------
    # Build model (GRU or LSTM)
    # ---------------------------
    model_type = "gru"     # Change to "lstm" if needed
    model = get_model(model_type, (X.shape[1], X.shape[2]))

    # ---------------------------
    # Train model
    # ---------------------------
    history = train_model(model, X, y)

    # Plot training loss
    plot_loss(history)

    # ---------------------------
    # Predict future cases
    # ---------------------------
    future = predict_future(model, X[-1], scaler, steps=10)

    print("\nPredicted COVID Cases (Next 10 Weeks):")
    print(future)

    # ---------------------------
    # Visualize Results
    # ---------------------------
    plot_results(df, country, future, scaler)


if __name__ == "__main__":
    main()


