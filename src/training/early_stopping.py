# Import required libraries.
import torch


class EarlyStopping:

    # Initialize early stopping.
    def __init__(
        self,
        patience,
        min_delta,
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")

        self.counter = 0

        self.should_stop = False

    # Check whether validation loss has improved.
    def __call__(self, validation_loss):

        if validation_loss < (self.best_loss - self.min_delta):

            self.best_loss = validation_loss

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.should_stop = True

        return self.should_stop