import matplotlib.pyplot as plt


class Metrics:
    def __init__(
        self,
        train_loss=[],
        train_accuracy=[],
        validation_loss=[],
        validation_accuracy=[]
    ) -> None:
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy

        self.validation_loss = validation_loss
        self.validation_accuracy = validation_accuracy

    def plot(self):
        if not (
            len(self.train_loss)
            == len(self.train_accuracy)
            == len(self.validation_loss)
            == len(self.validation_accuracy)
        ):
            raise ValueError("Metrics don't have the same length.")

        if len(self.train_loss) == 0:
            raise ValueError("Metrics can't me empty.")

        _, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot train and validation loss
        axes[0].plot(self.train_loss, label='Train Loss', )
        axes[0].plot(self.validation_loss, label='Validation Loss')
        axes[0].set_title('Train and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle='--')

        # Plot train and validation accuracy
        axes[1].plot(self.train_accuracy, label='Train Accuracy')
        axes[1].plot(self.validation_accuracy, label='Validation Accuracy')
        axes[1].set_title('Train and Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, linestyle='--')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plots
        plt.show()
