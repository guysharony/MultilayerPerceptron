import numpy as np
from src.multilayer_perceptron.Metrics import Metrics


class MultilayerPerceptron:
    def __init__(
        self,
        layers: list[int],
        epochs: int,
        loss: str,
        batch_size: int,
        learning_rate: float,
        weights=[],
        bias=[],
        activations=[]
    ) -> None:
        """
        Initializes the Multilayer Perceptron class with specified parameters.

        Parameters:
        - layers (list[int]): Number of neurons in each layer of the network.
        - epochs (int): Number of times the entire dataset is passed through
            the network.
        - loss (str): Type of loss function to use (not dynamically utilized
            in this implementation).
        - batch_size (int): Number of samples per batch to pass through the
            network.
        - learning_rate (float): Step size at each iteration while moving
            toward a minimum of the loss function.
        """
        self.layers = layers
        self.epochs = epochs
        self.loss = loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.weights = weights
        self.bias = bias
        self.activations = activations

        self.metrics = None

    def initialize_network(self):
        """
        Initializes weights, biases, and activation functions for the network.
        Weights and biases are initialized with random values.
        """
        for i in range(len(self.layers) - 1):
            self.weights.append(
                np.random.randn(
                    self.layers[i + 1],
                    self.layers[i]
                )
            )
            self.bias.append(
                np.random.randn(
                    self.layers[i + 1],
                    1
                )
            )
            self.activations.append(
                'softmax' if i == len(self.layers) - 2 else 'sigmoid'
            )

    def propagate_forward(self, x):
        """
        Forward propagates the input x through the network and returns
        all layer activations.

        Parameters:
        - x (numpy.ndarray): Input data.

        Returns:
        - list[numpy.ndarray]: Activations from all layers including
            input layer.
        """
        activations = [x.T]
        for i in range(len(self.layers) - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.bias[i]
            if self.activations[i] == 'sigmoid':
                activations.append(1 / (1 + np.exp(-z)))
            elif self.activations[i] == 'softmax':
                exp_z = np.exp(z - np.max(z, axis=0))
                activations.append(exp_z / np.sum(exp_z, axis=0))
            else:
                raise ValueError(
                    "Unknown activation function: {}".format(
                        self.activations[i]
                    )
                )
        return activations

    def propagate_backward(self, x, y, activations):
        """
        Backpropagates the error through the network and updates weights
        and biases.

        Parameters:
        - x (numpy.ndarray): Input data.
        - y (numpy.ndarray): True labels.
        - activations (list[numpy.ndarray]): Activations from all layers
            as returned by forward propagation.
        """
        m = x.shape[0]
        output_error = 2 * (activations[-1] - y.T)
        for i in reversed(range(len(self.layers) - 1)):
            delta = output_error * (
                activations[i + 1] * (1 - activations[i + 1])
            )
            grad_weights = np.dot(delta, activations[i].T) / m
            grad_bias = np.sum(delta, axis=1, keepdims=True) / m
            self.weights[i] -= self.learning_rate * grad_weights
            self.bias[i] -= self.learning_rate * grad_bias
            if i != 0:
                output_error = np.dot(self.weights[i].T, delta)

    def compute_loss(self, y_true, y_pred):
        """
        Computes the binary cross-entropy loss between predicted
        and true labels.

        Parameters:
        - y_true (numpy.ndarray): True labels.
        - y_pred (numpy.ndarray): Predicted labels/output of the
            network.

        Returns:
        - float: Computed binary cross-entropy loss.
        """
        y_pred = np.clip(y_pred.T, 1e-15, 1 - 1e-15)
        return np.mean(
            - (
                y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            )
        )

    def predict(self, x):
        """
        Predicts the output for given input x using forward propagation.

        Parameters:
        - x (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Predicted labels/output of the network.
        """
        return self.propagate_forward(x)[-1].T

    def fit(self, x, y, train=True):
        """
        Fits the model to the data, using specified batch size and
        updating the model for each batch.

        Parameters:
        - x (numpy.ndarray): Input data.
        - y (numpy.ndarray): True labels.

        Returns:
        - float: Total loss after the epoch.
        """
        loss = 0
        for i in range(0, x.shape[0], self.batch_size):
            x_batch = x[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]
            activations = self.propagate_forward(x_batch)
            loss += self.compute_loss(
                y_batch, activations[-1]
            ) * x_batch.shape[0]
            if train is True:
                self.propagate_backward(
                    x_batch,
                    y_batch,
                    activations
                )
        return loss / x.shape[0]

    def accuracy(self, x, y):
        """
        Calculates the accuracy of the model on provided data.

        Parameters:
        - x (numpy.ndarray): Input data.
        - y (numpy.ndarray): True labels.

        Returns:
        - float: Accuracy as a percentage of correctly predicted
            labels.
        """
        return np.mean(
            np.argmax(
                self.predict(x), axis=1
            ) == np.argmax(y, axis=1)
        )

    def train(self, training_data, validation_data):
        """
        Trains the model using the training data and evaluates
        on the validation data after each epoch.

        Parameters:
        - training_data (tuple(numpy.ndarray, numpy.ndarray)):
            Training data and labels.
        - validation_data (tuple(numpy.ndarray, numpy.ndarray)):
            Validation data and labels.
        """
        x_train, y_train = training_data

        print(f'x_train shape : {x_train.shape}')
        print(f'y_train shape : {y_train.shape}')

        self.initialize_network()

        train_losses = []
        validation_losses = []

        train_accuracies = []
        validation_accuracies = []

        for epoch in range(self.epochs):
            train_loss = self.fit(x_train, y_train)
            train_accuracy = self.accuracy(x_train, y_train)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            output = f'epoch {epoch + 1}/{self.epochs}'

            if validation_data:
                x_validation, y_validation = validation_data

                validation_loss = self.fit(x_validation, y_validation, False)
                validation_accuracy = self.accuracy(x_validation, y_validation)

                validation_losses.append(validation_loss)
                validation_accuracies.append(validation_accuracy)

                output += f" - loss: {train_loss:.4f}"
                output += f' - val_loss: {validation_loss:.4f}'
                output += f' - acc: {train_accuracy:.4f}'
                output += f' - val_acc: {validation_accuracy:.4f}'
            else:
                output += f' - loss: {train_loss:.4f}'
                output += f' - acc: {train_accuracy:.4f}'

            print(output)

        self.metrics = Metrics(
            train_losses,
            train_accuracies,
            validation_losses,
            validation_accuracies
        )

    def save(self, destination: str, x_min: float, x_max: float):
        data = {
            'layers': self.layers,
            'epochs': self.epochs,
            'loss': self.loss,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weights': self.weights,
            'bias': self.bias,
            'activations': self.activations,
            'x_min': x_min,
            'x_max': x_max
        }
        np.save(destination, np.array(data, dtype=object))
        print(f"> saving model '{destination}' to disk...")
