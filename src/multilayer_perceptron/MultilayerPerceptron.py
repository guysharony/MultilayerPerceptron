import numpy as np

class MultilayerPerceptron:
    def __init__(
        self,
        layers: list[int],
        epochs: int,
        loss: str,
        batch_size: int,
        learning_rate: float
    ) -> None:
        # Initializing variables
        self.layers = layers
        self.total_layers = len(self.layers)
        self.epochs = epochs
        self.loss = loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.weights = []
        self.bias = []
        self.activations = []

    def _initialize(self, x, y):
        for i in range(self.total_layers - 1):
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
                'softmax' if i == self.total_layers - 2 else 'sigmoid'
            )

            print(f'W {i}: ', self.weights[i].shape)
            print(f'B {i}: ', self.bias[i].shape)
            print(f'A {i}: ', self.activations[i])
            print()

    def _create_batch(self, x, y, i):
        """ Create batches for gradient descent to train on

        Args:
            x (np.ndarray): input features
            y (np.ndarray): target labels
            i (int): current iteration index

        Returns:
            tuple: batched x and y 
        """
        m, _ = x.shape

        if self.batch_size is None: # Batch gradient descent
            x_batch = x
            y_batch = y
        elif self.batch_size == 1: # Stochastic gradient descent
            random_index = np.random.randint(0, m)
            x_batch = x[random_index]
            y_batch = y[random_index]
        else: # Mini-batch gradient descent
            np.random.seed(i)

            indices = np.arange(m)
            np.random.shuffle(indices)

            x_shuffled = x[indices]
            y_shuffled = y[indices]

            x_batch = x_shuffled[:self.batch_size]
            y_batch = y_shuffled[:self.batch_size]

        return x_batch, y_batch

    def forward(self, x):
        a = [x.T]

        for i in range(self.total_layers - 1):
            activation = self.activations[i]
            weights = self.weights[i]
            bias = self.bias[i]

            z = np.dot(weights, a[-1]) + bias
            if activation == 'sigmoid':
                a_output = 1 / (1 + np.exp(-z))
            elif activation == 'softmax':
                exp_sum = np.exp(z)
                a_output = exp_sum / np.sum(exp_sum, axis=0)
            else:
                raise ValueError("Unknown activation function: {}".format(activation))

            a.append(a_output)

        return a

    def backward(self, x, y, a):
        m = x.shape[0]
        dCA = 2 * (a[-1] - y.T)
        for i in reversed(range(self.total_layers - 1)):
            dZA = a[i + 1] * (1 - a[i + 1])
            dAZ_CA = dCA * dZA
            dW = np.dot(dAZ_CA, a[i].T) / m
            dB = np.sum(dAZ_CA) / m

            self.weights[i] -= self.learning_rate * dW
            self.bias[i] -= self.learning_rate * dB

            dCA = np.dot(self.weights[i].T, dAZ_CA)

    def compute_loss(self, y_true, y_prediction):
        y_prediction = np.clip(y_prediction.T, 1e-15, 1 - 1e-15)

        return np.mean(
            - (
                y_true * np.log(y_prediction) + (1 - y_true) * np.log(1 - y_prediction)
            )
        )

    # def predict(self, x):
    #    a = self.forward(x)
    #    return a[-1].T

    # def accuracy(self, y_true, y_pred):
        # Calculate accuracy by comparing predicted classes to ground truth classes
    #    y_pred_classes = np.argmax(y_pred, axis=1)
    #    y_true_classes = np.argmax(y_true, axis=1)
    #    return np.mean(y_pred_classes == y_true_classes)

    def fit(self, training, validation):
        x = training[0]
        y = training[1]

        self._initialize(x, y)

        for epoch in range(self.epochs):
            # total_loss = 0.0
            # correct_predictions = 0
            for i in range(0, x.shape[0], self.batch_size):
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                a = self.forward(x_batch)

                # loss = self.compute_loss(y_batch, a[-1])
                # total_loss += loss * x_batch.shape[0]

                self.backward(x_batch, y_batch, a)

                # batch_predictions = self.predict(x_batch)
                # batch_correct = np.sum(np.argmax(batch_predictions, axis=0) == np.argmax(y_batch, axis=0))
                # correct_predictions += batch_correct

            # train_predictions = self.predict(x)
            # train_accuracy = self.accuracy(y, train_predictions)

            # print(train_accuracy)