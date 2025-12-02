from tensorflow.keras.callbacks import EarlyStopping
from src.model_lstm import build_lstm
from src.model_gru import build_gru


def get_model(model_type, input_shape):
    """
    Returns either an LSTM or GRU model based on model_type.
    """
    if model_type.lower() == "gru":
        return build_gru(input_shape)
    else:
        return build_lstm(input_shape)


def train_model(model, X, y):
    """
    Train the passed model (LSTM or GRU) on data.
    Adds validation split and early stopping.
    """

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X, y,
        epochs=20,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    return history


