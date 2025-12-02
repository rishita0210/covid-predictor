import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
