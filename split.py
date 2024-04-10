import sys

from src.split.dataset import load_dataset
from src.split.dataset import add_column_names

from src.split.plot import plot


def main():
    if len(sys.argv) != 2:
        raise ValueError("split.py [path to dataset]")

    # Loading dataset
    dataset = load_dataset(sys.argv[1])

    # Loading column names
    dataset = add_column_names(dataset)

    # Represent data with graphics
    print(dataset)

    # Création du diagramme de dispersion
    plot(dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f'error: {error}')
