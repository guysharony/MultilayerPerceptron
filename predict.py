from src.predict.arguments import arguments

from src.multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron


def main():
    """
    Main function to load a pre-trained model and make predictions on a
    dataset.
    """

    # Loading arguments
    args = arguments()

    model = MultilayerPerceptron.load(args.model)
    model.predict(args.dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}")
