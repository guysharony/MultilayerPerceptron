import pandas as pd


def load_dataset(validation_path):
    # Loading validation data
    validation = pd.read_csv(validation_path)
    x_validation = validation.drop("Diagnosis", axis=1)
    y_validation = validation["Diagnosis"]

    # Normalizing validation
    validation_dataset = (x_validation, y_validation)

    return validation_dataset
