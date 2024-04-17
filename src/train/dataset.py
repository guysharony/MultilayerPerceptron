import numpy as np
import pandas as pd

from src.normalize import normalize


def normalize_dataset(x, y, minimum=None, maximum=None):
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


def load_dataset(training_path, validation_path):
    # Loading training data
    training = pd.read_csv(training_path)
    x_train = training.drop("Diagnosis", axis=1)
    y_train = training["Diagnosis"]

    # Loading validation data
    validation = pd.read_csv(validation_path)
    x_validation = validation.drop("Diagnosis", axis=1)
    y_validation = validation["Diagnosis"]

    # Normalizing training
    (
        training_dataset,
        x_min,
        x_max
    ) = normalize_dataset(
        x_train,
        y_train
    )

    # Normalizing validation
    (
        validation_dataset,
        _,
        _
    ) = normalize_dataset(
        x_validation,
        y_validation,
        x_min,
        x_max
    )

    return (
        training_dataset,
        validation_dataset,
        x_min,
        x_max
    )
