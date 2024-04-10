from src.train.arguments import arguments


def main():
    # Loading arguments
    args = arguments()

    # Loading training data
    

    print(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}")
