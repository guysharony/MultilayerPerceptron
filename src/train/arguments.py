import argparse
import numpy as np

from src.train.dataset import load_dataset


def arguments():
    parser = argparse.ArgumentParser(
        description="""
            A multilayer perceptron training program.
        """
    )

    parser.add_argument(
        "-t", "--training_dataset",
        type=str,
        help="Path to the training dataset.",
        default="./datasets/train.csv"
    )

    parser.add_argument(
        "-v", "--validation_dataset",
        type=str,
        help="Path to the training dataset.",
        default="./datasets/validation.csv"
    )

    parser.add_argument(
        "-s", "--save",
        type=str,
        help="Path to model saving file.",
        default="./saved_model.npy"
    )

    parser.add_argument(
        "-l", "--layers",
        type=int,
        nargs="+",
        help="The layers of the each neuron.",
        default=[24, 24]
    )

    parser.add_argument(
        "-e", "--epochs",
        type=int,
        help="The number of epochs.",
        default=100
    )

    parser.add_argument(
        "--loss",
        type=str,
        help="The loss function to use.",
        default="binaryCrossentropy"
    )

    parser.add_argument(
        "-bs", "--batch_size",
        type=int,
        help="The batch size to use.",
        default=64
    )

    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        help="The learning rate of the neural network.",
        default=0.001
    )

    args = parser.parse_args()

    # Loading dataset
    (
        training_dataset,
        validation_dataset,
    ) = load_dataset(
        args.training_dataset,
        args.validation_dataset
    )

    args.training_dataset = training_dataset
    args.validation_dataset = validation_dataset

    # Computing outputs
    outputs = len(np.unique(training_dataset[1].to_numpy()))

    # Adding inputs and outputs to layers
    args.layers = [training_dataset[0].shape[1]] + args.layers
    args.layers = args.layers + [outputs]

    return args
