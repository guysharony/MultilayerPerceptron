import numpy as np
import pandas as pd

from src.normalize import normalize


def normalize_dataset(dataset):
    x = dataset.drop("Diagnosis", axis=1)
    y = dataset["Diagnosis"]

    x_minimum = x.min()
    x_maximum = x.max()

    # Normalizing x features
    x_normalized = normalize(x, x_minimum, x_maximum)
    x_normalized = x_normalized.to_numpy()

    y = y.map({
        'M': [0, 1],
        'B': [1, 0]
    })
    y = np.asarray(y.tolist())

    return x_normalized, y


def load_dataset(training_path, validation_path):
    # Loading dataset
    training = pd.read_csv(training_path)
    validation = pd.read_csv(validation_path)

    training_dataset = normalize_dataset(training)
    validation_dataset = normalize_dataset(validation)

    return training_dataset, validation_dataset
