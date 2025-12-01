from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model():
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1)),
        LSTM(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model
