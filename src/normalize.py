import numpy as np


def normalize(feature, minimum, maximum):
    return (feature - minimum) / (maximum - minimum)


def normalize_dataset(dataset, minimum=None, maximum=None):
    x, y = dataset

    x_minimum = x.min() if minimum is None else minimum
    x_maximum = x.max() if maximum is None else maximum

    # Normalizing x features
    x_normalized = normalize(x, x_minimum, x_maximum)
    x_normalized = x_normalized.to_numpy()

    y = y.map({
        'M': [0, 1],
        'B': [1, 0]
    })
    y = np.asarray(y.tolist())

    return (x_normalized, y), x_minimum, x_maximum
