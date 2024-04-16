from src.train.arguments import arguments

from src.train.dataset import load_dataset

from src.multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron
from src.train.plot import plot_metrics


def main():
    # Loading arguments
    args = arguments()

    # Loading datasets
    training_dataset, validation_dataset = load_dataset(
        args.training_dataset,
        args.validation_dataset
    )

    # Adding input and output layers
    args.layers = [training_dataset[0].shape[1]] + args.layers
    args.layers = args.layers + [training_dataset[1].shape[1]]

    # Creating model
    model = MultilayerPerceptron(
        args.layers,
        args.epochs,
        args.loss,
        args.batch_size,
        args.learning_rate
    )
    metrics = model.train(
        training_dataset,
        validation_dataset
    )
    plot_metrics(metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}")
