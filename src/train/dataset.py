import pandas as pd

from src.columns import add_column_names


def load_dataset(training_path, validation_path):
    """
    Loads and processes the training and validation datasets from CSV files.

    Args:
        training_path (str): The file path to the training dataset CSV.
        validation_path (str): The file path to the validation dataset CSV.

    Returns:
        tuple: A tuple containing:
            - training_dataset (tuple): A tuple with features and
                labels from the training dataset.
            - validation_dataset (tuple): A tuple with features and
                labels from the validation dataset.
    """
    # Loading training data
    training = pd.read_csv(training_path)

    # Loading column names
    training = add_column_names(training)

    training_dataset = (
        training.drop("Diagnosis", axis=1),
        training["Diagnosis"]
    )

    # Loading validation data
    validation = pd.read_csv(validation_path)

    # Loading column names
    validation = add_column_names(validation)

    validation_dataset = (
        validation.drop("Diagnosis", axis=1),
        validation["Diagnosis"]
    )

    return (
        training_dataset,
        validation_dataset,
    )
