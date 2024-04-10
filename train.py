from src.train.arguments import arguments

from src.train.dataset import load_dataset


def main():
    # Loading arguments
    args = arguments()

    # Loading datasets
    training_dataset, validation_dataset = load_dataset(
        args.training_dataset,
        args.validation_dataset
    )

    print(training_dataset)
    print(validation_dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}")
