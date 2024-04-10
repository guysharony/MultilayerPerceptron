import numpy as np
import pandas as pd


def load_dataset(arg):
    return pd.read_csv(arg, header=None)


def add_column_names(dataset):
    column_names = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave points",
        "symmetry",
        "fractal dimension"
    ]

    mean_columns = [
        f"mean {column_name}"
        for column_name in column_names
    ]

    standard_error_columns = [
        f"standard error {column_name}"
        for column_name in column_names
    ]

    largest_columns = [
        f"largest {column_name}"
        for column_name in column_names
    ]

    dataset.columns = (
        [
            "ID number",
            "Diagnosis"
        ] +
        mean_columns +
        standard_error_columns +
        largest_columns
    )

    return dataset


def split_dataset(dataset, proportion):
    # Shuffle dataset
    shuffled_dataset = dataset.sample(frac=1)

    # Computinf split index
    split_index = int(shuffled_dataset.shape[0] * proportion)

    # Spliting dataset
    train = shuffled_dataset[:split_index]
    validation = shuffled_dataset[split_index:]

    return train, validation