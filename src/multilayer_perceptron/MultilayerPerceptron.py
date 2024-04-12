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
        self.hidden_layers = len(self.layers)
        self.total_layers = self.hidden_layers + 1
        self.epochs = epochs
        self.loss = loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.weights = []
        self.bias = []
        self.activations = []

    def _initialize(self, x, y):
        n_features = x.shape[1]

        for i in range(self.hidden_layers):
            self.weights.append(
                np.random.randn(
                    self.layers[i],
                    n_features if i == 0 else self.layers[i - 1]
                )
            )
            self.bias.append(
                np.random.randn(
                    self.layers[i],
                    1
                )
            )
            self.activations.append(
                'sigmoid'
            )

        self.weights.append(
            np.random.randn(
                y.shape[1],
                self.layers[-1]
            )
        )
        self.bias.append(
            np.random.randn(
                y.shape[1],
                1
            )
        )

        self.activations.append(
            'softmax'
        )

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
        output_layers = [x.T]

        for i in range(self.total_layers):
            activation = self.activations[i]
            weights = self.weights[i]
            bias = self.bias[i]

            weighted_sum = np.dot(weights, output_layers[-1]) + bias
            if activation == 'sigmoid':
                activated_output = 1 / (1 + np.exp(-weighted_sum))
            elif activation == 'softmax':
                exp_sum = np.exp(weighted_sum)
                activated_output = exp_sum / np.sum(exp_sum, axis=0)
            else:
                raise ValueError("Unknown activation function: {}".format(activation))

            output_layers.append(activated_output)

        return output_layers

    def compute_loss(self, y_true, y_prediction):
        y_prediction = np.clip(y_prediction.T, 1e-15, 1 - 1e-15)

        return np.mean(
            - (
                y_true * np.log(y_prediction) + (1 - y_true) * np.log(1 - y_prediction)
            )
        )

    def fit(self, training, validation):
        x = training[0]
        y = training[1]

        self._initialize(x, y)

        for epoch in range(self.epochs):
            x_batch, y_batch = self._create_batch(x, y, epoch)

            output = self.forward(x_batch)

            loss = self.compute_loss(y_batch, output[-1])

            print(loss)
