import sys

from src.columns import add_column_names
from src.split.dataset import load_dataset
from src.split.dataset import split_dataset

from src.split.plot import plot


def main():
    """
    Main function to load, process, and split a dataset.

    Raises:
        ValueError: If the number of command-line arguments is not equal to 2.
    """

    if len(sys.argv) != 2:
        raise ValueError("python split.py [path to dataset]")

    # Loading dataset
    dataset = load_dataset(sys.argv[1])

    # Loading column names
    dataset = add_column_names(dataset)

    # Creating a diagram
    plot(dataset)

    # Split dataset
    train, validation = split_dataset(dataset, 0.8)

    # Saving the splited dataset
    train.to_csv('datasets/train.csv', index=False, header=False)
    validation.to_csv('datasets/validation.csv', index=False, header=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f'error: {error}')
