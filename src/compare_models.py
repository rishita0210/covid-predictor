import matplotlib.pyplot as plt
from src.model_lstm import build_lstm
from src.model_gru import build_gru
from tensorflow.keras.callbacks import EarlyStopping

def train_and_compare(X, y):
    input_shape = (X.shape[1], X.shape[2])
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Train LSTM
    lstm = build_lstm(input_shape)
    history_lstm = lstm.fit(
        X, y, 
        epochs=20, 
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Train GRU
    gru = build_gru(input_shape)
    history_gru = gru.fit(
        X, y, 
        epochs=20, 
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.plot(history_lstm.history["val_loss"], label="LSTM Val Loss")
    plt.plot(history_gru.history["val_loss"], label="GRU Val Loss")
    plt.title("LSTM vs GRU Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return lstm, gru, history_lstm, history_gru

