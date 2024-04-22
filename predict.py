from src.predict.arguments import arguments

from src.multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron


def main():
    # Loading arguments
    args = arguments()

    model = MultilayerPerceptron.load(args.model)
    model.predict(args.dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}")
