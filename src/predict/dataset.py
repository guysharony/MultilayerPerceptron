import pandas as pd


def load_dataset(validation_path):
    """
    Loads and processes the validation dataset from a CSV file.

    Args:
        validation_path (str): The file path to the validation dataset CSV.

    Returns:
        tuple: A tuple containing:
            - x_validation (DataFrame): Features from the validation
                dataset.
            - y_validation (Series): Labels ('Diagnosis') from the validation
                dataset.
    """
    # Loading validation data
    validation = pd.read_csv(validation_path)
    x_validation = validation.drop("Diagnosis", axis=1)
    y_validation = validation["Diagnosis"]

    # Normalizing validation
    validation_dataset = (x_validation, y_validation)

    return validation_dataset
