import argparse

from src.predict.dataset import load_dataset


def arguments():
    parser = argparse.ArgumentParser(
        description="""
            A multilayer perceptron prediction program.
        """
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        help="Path to the training dataset.",
        default="./datasets/validation.csv"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Path to model to load.",
        default="./saved_model.npy"
    )

    args = parser.parse_args()

    # Loading dataset
    args.dataset = load_dataset(
        args.dataset
    )

    return args
