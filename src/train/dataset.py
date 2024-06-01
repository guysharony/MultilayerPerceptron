import pandas as pd

from src.columns import add_column_names

def load_csv(dataset):
    try:
        return pd.read_csv(dataset)
    except Exception as e:
        return None

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
    training = load_csv(training_path)
    if training is None:
        raise ValueError(
            f'training_dataset ({training_path}) is not valid.'
        )

    # Loading column names
    training = add_column_names(training)

    training_dataset = (
        training.drop("Diagnosis", axis=1),
        training["Diagnosis"]
    )

    # Loading validation data
    validation = load_csv(validation_path)
    if validation is None:
        validation_dataset = None
    else:
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
