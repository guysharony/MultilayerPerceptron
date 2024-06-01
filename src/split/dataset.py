import pandas as pd


def load_dataset(arg):
    """
    Load a dataset from a CSV file without headers.

    Args:
        arg (str): The file path to the CSV dataset.

    Returns:
        DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(arg, header=None)


def split_dataset(dataset, proportion):
    """
    Splits a dataset into training and validation sets based on a
    specified proportion.

    Args:
        dataset (DataFrame): The dataset to split.
        proportion (float): The proportion of the dataset to use
            as the training set.

    Returns:
        tuple: A tuple containing:
            - train (DataFrame): The training subset of the dataset.
            - validation (DataFrame): The validation subset of the dataset.
    """
    # Shuffle dataset
    shuffled_dataset = dataset.sample(frac=1)

    # Computinf split index
    split_index = int(shuffled_dataset.shape[0] * proportion)

    # Spliting dataset
    train = shuffled_dataset[:split_index]
    validation = shuffled_dataset[split_index:]

    return train, validation
