import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_results(df, country, future_predictions, scaler=None):
    # Filter the selected country
    df_country = df[df["Entity"] == country].copy()
    df_country["Day"] = pd.to_datetime(df_country["Day"])
    df_country = df_country.sort_values("Day")

    # Actual weekly cases
    actual = df_country["Weekly cases"].values

    # If predictions were scaled, inverse transform them
    if scaler is not None:
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_predictions = future_predictions.flatten()
    
    # Ensure predictions are non-negative
    future_predictions = np.maximum(future_predictions, 0)

    # Build future dates
    last_date = df_country["Day"].iloc[-1]
    future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(len(future_predictions))]

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(df_country["Day"], actual, label="Actual Weekly Cases")
    plt.plot(future_dates, future_predictions, label="Predicted Future Cases", linestyle="--", color='red')

    plt.xlabel("Date")
    plt.ylabel("Weekly Cases")
    plt.title(f"COVID-19 Weekly Cases Prediction for {country}")
    plt.legend()
    plt.grid(True)
    plt.show()

