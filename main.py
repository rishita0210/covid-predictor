from src.load_data import load_covid_data
from src.preprocess import preprocess
from src.compare_models import train_and_compare
from src.predict_future import predict_future
from src.plot_loss import plot_loss
from src.visualize import plot_results
from src.compare_predictions import compare_predictions


def main():
    df = load_covid_data()
    country = "India"

    X, y, scaler = preprocess(df, country)

    # 1) Train both models & compare loss
    lstm, gru, history_lstm, history_gru = train_and_compare(X, y)

    # 2) Individual loss curves
    print("\nPlotting LSTM Loss Curve…")
    plot_loss(history_lstm)

    print("\nPlotting GRU Loss Curve…")
    plot_loss(history_gru)

    # 3) Predict future using both models
    lstm_future = predict_future(lstm, X[-1], scaler, steps=10)
    gru_future = predict_future(gru, X[-1], scaler, steps=10)

    print("\nLSTM Predicted Future Cases:")
    print(lstm_future)

    print("\nGRU Predicted Future Cases:")
    print(gru_future)

    # 4) Plot prediction comparison graph
    compare_predictions(lstm_future, gru_future)

    # 5) Plot final graph (GRU or LSTM)
    plot_results(df, country, gru_future, scaler)  # using GRU but you can change

if __name__ == "__main__":
    main()


