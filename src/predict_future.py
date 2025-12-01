import numpy as np
import matplotlib.pyplot as plt

def predict_future(model, last_seq, scaler):
    future = []
    
    seq = last_seq.copy()

    for _ in range(30):
        pred = model.predict(seq.reshape(1,30,1))
        future.append(pred[0,0])
        seq = np.vstack([seq[1:], pred])

    future = scaler.inverse_transform(np.array(future).reshape(-1,1))

    plt.plot(future)
    plt.title("Next 30 Days Forecast")
    plt.savefig("plots/future_forecast.png")
    plt.close()

    return future
