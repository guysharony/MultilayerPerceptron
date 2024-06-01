import matplotlib.pyplot as plt


class Metrics:
    """
    A class for storing and visualizing training and validation metrics.
    """
    def __init__(
        self,
        train_loss=[],
        train_accuracy=[],
        validation_loss=[],
        validation_accuracy=[]
    ) -> None:
        """
        Initializes the Metrics class with lists to store loss and accuracy
        for training and validation.

        Args:
            train_loss (list, optional): Initial list of training loss
                values. Default is an empty list.
            train_accuracy (list, optional): Initial list of training
                accuracy values. Default is an empty list.
            validation_loss (list, optional): Initial list of validation
                loss values. Default is an empty list.
            validation_accuracy (list, optional): Initial list of validation
                accuracy values. Default is an empty list.
        """
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy

        self.validation_loss = validation_loss
        self.validation_accuracy = validation_accuracy

    def plot(self):
        """
        Plots the training and validation loss and accuracy over epochs.

        Raises:
            ValueError: If the length of any metric list does not match
                the others, or if they are empty.
        """
        if not (
            len(self.train_loss) == len(self.train_accuracy)
            and (
                self.validation_loss is None
                or len(self.validation_loss) == len(self.train_loss)
            )
            and (
                self.validation_accuracy is None
                or len(self.validation_accuracy) == len(self.train_accuracy)
            )
        ):
            raise ValueError("Metrics don't have the same length.")

        if len(self.train_loss) == 0:
            raise ValueError("Metrics can't me empty.")

        _, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot train and validation loss
        axes[0].plot(self.train_loss, label='Train Loss', )
        if self.validation_loss is not None:
            axes[0].plot(self.validation_loss, label='Validation Loss')
        axes[0].set_title('Train and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle='--')

        # Plot train and validation accuracy
        axes[1].plot(self.train_accuracy, label='Train Accuracy')
        if self.validation_accuracy is not None:
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
