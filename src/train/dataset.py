import pandas as pd


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
    training_dataset = (
        training.drop("Diagnosis", axis=1),
        training["Diagnosis"]
    )

    # Loading validation data
    validation = pd.read_csv(validation_path)
    validation_dataset = (
        validation.drop("Diagnosis", axis=1),
        validation["Diagnosis"]
    )

    return (
        training_dataset,
        validation_dataset,
    )
