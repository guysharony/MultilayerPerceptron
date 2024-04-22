import pandas as pd


def load_dataset(training_path, validation_path):
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
