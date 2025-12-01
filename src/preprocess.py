import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, country="India"):
    # Filter data by country
    df = df[df["Entity"] == country]

    # Convert date column
    df["Day"] = pd.to_datetime(df["Day"])
    df = df.sort_values("Day")

    # Extract weekly cases
    cases = df["Weekly cases"].astype(float).values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_cases = scaler.fit_transform(cases)

    # Create sequences (LSTM input)
    sequence_length = 10
    X, y = [], []

    for i in range(len(scaled_cases) - sequence_length):
        X.append(scaled_cases[i:i + sequence_length])
        y.append(scaled_cases[i + sequence_length])

    return np.array(X), np.array(y), scaler
