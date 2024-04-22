from src.predict.dataset import load_dataset
from src.multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron


def main():
    validation_dataset = load_dataset('./datasets/validation.csv')

    model = MultilayerPerceptron.load('./saved_model.npy')
    model.predict(validation_dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}")
