import numpy as np

def predict_future(model, last_seq, scaler, steps=10):
    predictions = []
    seq = last_seq.copy()

    for _ in range(steps):
        pred = model.predict(seq.reshape(1, seq.shape[0], seq.shape[1]))[0][0]
        predictions.append(pred)

        seq = np.vstack([seq[1:], [pred]])

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions
