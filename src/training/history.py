# Import required libraries.
from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    """
    Store training statistics collected during each epoch.
    """

    # Store epoch-wise training loss.
    train_losses: list[float] = field(default_factory=list)

    # Store epoch-wise validation loss.
    validation_losses: list[float] = field(default_factory=list)

    # Store epoch-wise training accuracy.
    train_accuracies: list[float] = field(default_factory=list)

    # Store epoch-wise validation accuracy.
    validation_accuracies: list[float] = field(default_factory=list)

    # Store learning rate history.
    learning_rates: list[float] = field(default_factory=list)

    # Store epoch execution time.
    epoch_times: list[float] = field(default_factory=list)

    # Add one epoch of results.
    def update(
        self,
        train_loss: float,
        validation_loss: float,
        train_accuracy: float,
        validation_accuracy: float,
        learning_rate: float,
        epoch_time: float,
    ) -> None:

        self.train_losses.append(train_loss)

        self.validation_losses.append(validation_loss)

        self.train_accuracies.append(train_accuracy)

        self.validation_accuracies.append(validation_accuracy)

        self.learning_rates.append(learning_rate)

        self.epoch_times.append(epoch_time)

    # Convert history into a dictionary.
    def to_dict(self) -> dict:

        return {

            "train_losses": self.train_losses,

            "validation_losses": self.validation_losses,

            "train_accuracies": self.train_accuracies,

            "validation_accuracies": self.validation_accuracies,

            "learning_rates": self.learning_rates,

            "epoch_times": self.epoch_times,
        }