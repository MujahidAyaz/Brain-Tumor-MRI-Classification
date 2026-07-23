# Import required libraries.
from dataclasses import dataclass, field


@dataclass
class TrainingHistory:

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

    # Add one epoch to history.
    def update(
        self,
        train_loss,
        validation_loss,
        train_accuracy,
        validation_accuracy,
        learning_rate,
        epoch_time,
    ):

        self.train_losses.append(train_loss)
        self.validation_losses.append(validation_loss)

        self.train_accuracies.append(train_accuracy)
        self.validation_accuracies.append(validation_accuracy)

        self.learning_rates.append(learning_rate)

        self.epoch_times.append(epoch_time)