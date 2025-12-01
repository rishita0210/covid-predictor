def train_model(model, X, y):
    history = model.fit(
        X, y,
        epochs=20,
        batch_size=16,
        validation_split=0.2
    )
    return history

