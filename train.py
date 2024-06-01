from src.train.arguments import arguments

from src.multilayer_perceptron.Metrics import Metrics
from src.multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron


def main():
    """
    Main function to create, train, and save a machine learning model.
    """

    # Loading arguments
    args = arguments()

    # Creating model
    model = MultilayerPerceptron(
        args.layers,
        args.epochs,
        args.batch_size,
        args.learning_rate
    )
    model.train(
        args.training_dataset,
        args.validation_dataset
    )
    model.save(args.save)

    if model.metrics.__class__ == Metrics:
        model.metrics.plot()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}")
