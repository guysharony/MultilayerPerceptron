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
        self.epochs = epochs
        self.loss = loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.weights = []
        self.bias = []
        self.activations = []

    def _initialize(self, x, y):
        n_features = x.shape[1]

        self.weights.append(
            np.random.randn(
                n_features,
                n_features
            )
        )

        self.bias.append(
            np.random.randn(
                n_features,
                1
            )
        )

        for i in range(self.hidden_layers):
            self.weights.append(
                np.random.randn(
                    self.layers[i],
                    self.layers[i - 1]
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
        self.activations.append(
            'softmax'
        )

        print(self.weights, n_features)

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
        output = [x.T]

        for i in range(len(self.layers) - 1):
            activation = self.activations[i]

            weigthed_sum = np.dot(self.weights[i], output[-1]) + self.bias[i]
            output.append((
                1 / (1 + np.exp(-weigthed_sum)) \
                    if activation == 'sigmoid' \
                    else np.exp(weigthed_sum) / (np.sum(np.exp(weigthed_sum), axis=0))
            ))

        return output

    def fit(self, training, validation):
        x = training[0]
        y = training[1]

        self._initialize(x, y)

        for epoch in range(self.epochs):
            x_batch, y_batch = self._create_batch(x, y, epoch)

            output = self.forward(x_batch)

            print('OUTPUT: ', output)
