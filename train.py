from src.train.arguments import arguments

from src.train.dataset import load_dataset

from src.multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron


def main():
    # Loading arguments
    args = arguments()

    # Loading datasets
    training_dataset, validation_dataset = load_dataset(
        args.training_dataset,
        args.validation_dataset
    )

    model = MultilayerPerceptron(
        args.layers,
        args.epochs,
        args.loss,
        args.batch_size,
        args.learning_rate
    )
    model.fit(
        training_dataset,
        validation_dataset
    )


if __name__ == "__main__":
    #try:
    main()
    #except Exception as error:
    #    print(f"error: {error}")
