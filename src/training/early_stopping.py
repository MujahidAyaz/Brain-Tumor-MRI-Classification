# Import required libraries.
from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """
    Stop training when the validation loss
    does not improve for a specified number of epochs.
    """

    patience: int
    min_delta: float

    best_loss: float = float("inf")

    counter: int = 0

    should_stop: bool = False

    # Check whether validation loss has improved.
    def __call__(
        self,
        validation_loss: float,
    ) -> bool:

        # Reset the counter if validation loss improves.
        if validation_loss < (self.best_loss - self.min_delta):

            self.best_loss = validation_loss

            self.counter = 0

            return False

        # Increase the counter if there is no improvement.
        self.counter += 1

        # Stop training after reaching patience.
        if self.counter >= self.patience:

            self.should_stop = True

        return self.should_stop

    # Reset the early stopping state.
    def reset(self) -> None:

        self.best_loss = float("inf")

        self.counter = 0

        self.should_stop = False