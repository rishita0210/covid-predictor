import matplotlib.pyplot as plt

def compare_predictions(lstm_preds, gru_preds):
    plt.figure(figsize=(8,5))
    plt.plot(lstm_preds, label="LSTM Predictions")
    plt.plot(gru_preds, label="GRU Predictions")
    plt.title("LSTM vs GRU Future Predictions")
    plt.xlabel("Future Steps")
    plt.ylabel("Predicted Cases")
    plt.grid(True)
    plt.legend()
    plt.show()
