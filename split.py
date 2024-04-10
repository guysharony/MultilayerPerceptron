import sys

from src.split.dataset import load_dataset
from src.split.dataset import add_column_names
from src.split.dataset import split_dataset

from src.split.plot import plot


def main():
    if len(sys.argv) != 2:
        raise ValueError("split.py [path to dataset]")

    # Loading dataset
    dataset = load_dataset(sys.argv[1])

    # Loading column names
    dataset = add_column_names(dataset)

    # Création du diagramme de dispersion
    plot(dataset)

    # Split dataset
    train, validation = split_dataset(dataset, 0.8)

    # Saving the splited dataset
    train.to_csv('datasets/train.csv', index=False)
    validation.to_csv('datasets/validation.csv', index=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f'error: {error}')
