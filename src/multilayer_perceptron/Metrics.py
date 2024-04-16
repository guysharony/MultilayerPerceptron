class Metrics:
    def __init__(
        self,
        train_loss=[],
        train_accuracy=[],
        validation_loss=[],
        validation_accuracy=[]
    ) -> None:
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy

        self.validation_loss = validation_loss
        self.validation_accuracy = validation_accuracy
