from src.load_data import load_covid_data
from src.preprocess import scale_data, create_sequences
from src.model_lstm import build_lstm_model
from src.train import train_model
from src.predict_future import predict_future
from src.visualize import plot_cases

# Load data
df = load_covid_data()
plot_cases(df)

# Preprocess
scaler, scaled = scale_data(df)
X, y = create_sequences(scaled)

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = build_lstm_model()

# Train
train_model(model, X_train, y_train, X_test, y_test, scaler)

# Save model
model.save("models/lstm_model.h5")

# Predict future
last_seq = scaled[-30:]
future = predict_future(model, last_seq, scaler)

print("Next 30 Days Prediction:")
print(future)
