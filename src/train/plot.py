import matplotlib.pyplot as plt

from src.multilayer_perceptron.Metrics import Metrics


def plot_metrics(metrics: Metrics):
    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot train and validation loss
    axes[0].plot(metrics.train_loss, label='Train Loss', )
    axes[0].plot(metrics.validation_loss, label='Validation Loss')
    axes[0].set_title('Train and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--')

    # Plot train and validation accuracy
    axes[1].plot(metrics.train_accuracy, label='Train Accuracy')
    axes[1].plot(metrics.validation_accuracy, label='Validation Accuracy')
    axes[1].set_title('Train and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()
