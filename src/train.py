import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, X_test, y_test, scaler):
    history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)

    pred = model.predict(X_test)
    pred_inv = scaler.inverse_transform(pred)
    y_test_inv = scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
    print("RMSE:", rmse)

    # Plot Train vs Test
    plt.figure(figsize=(10,5))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(pred_inv, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Cases")
    plt.savefig("plots/train_vs_test.png")
    plt.close()

    return history
