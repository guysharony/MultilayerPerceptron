import argparse


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
        default="./datasets/train.csv"
    )

    parser.add_argument(
        "-l", "--layers",
        type=int,
        nargs="+",
        help="The layers of the each neuron.",
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
        default=100
    )

    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        help="The learning rate of the neural network.",
        default=0.001
    )

    return parser.parse_args()
