import numpy as np


def normalize(feature, minimum, maximum):
    """
    Normalize a feature to a range between 0 and 1.

    Args:
        feature (float): The feature value to be normalized.
        minimum (float): The minimum value of the feature range.
        maximum (float): The maximum value of the feature range.

    Returns:
        float: The normalized feature value.
    """
    return (feature - minimum) / (maximum - minimum)


def normalize_dataset(dataset, minimum=None, maximum=None):
    """
    Normalizes the features in a dataset and encodes the target labels.

    Args:
        dataset (tuple): A tuple containing the feature DataFrame (x) and
            the target Series (y).
        minimum (float, optional): The minimum value for normalization.
            If None, the minimum value of x is used.
        maximum (float, optional): The maximum value for normalization.
            If None, the maximum value of x is used.

    Returns:
        tuple: A tuple containing:
            - Normalized features and encoded labels (x_normalized, y).
            - Minimum value used for normalization.
            - Maximum value used for normalization.
    """
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
