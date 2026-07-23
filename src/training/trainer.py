# Import required libraries.
import logging
import time

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    MIN_DELTA,
    MODEL_DIR,
)

from src.training.engine import Engine
from src.training.history import TrainingHistory
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import CheckpointManager


class Trainer:

    # Initialize trainer.
    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        device,
    ):

        self.model = model.to(device)

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.device = device

        # Configure loss function.
        self.criterion = nn.CrossEntropyLoss()

        # Configure optimizer.
        self.optimizer = Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
        )

        # Configure learning rate scheduler.
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        # Configure helper classes.
        self.engine = Engine(
            self.model,
            self.criterion,
            self.optimizer,
            self.device,
        )

        self.history = TrainingHistory()

        self.early_stopping = EarlyStopping(
            patience=PATIENCE,
            min_delta=MIN_DELTA,
        )

        self.checkpoint = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_path=MODEL_DIR / "last_checkpoint.pth",
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )

        self.logger = logging.getLogger(__name__)

    # Train the complete model.
    def train(self):

        start_epoch = 0

        best_validation_loss = float("inf")
        best_validation_accuracy = 0.0

        checkpoint = self.checkpoint.load()

        if checkpoint is not None:

            start_epoch = checkpoint["epoch"] + 1

            best_validation_loss = checkpoint["best_validation_loss"]

            best_validation_accuracy = checkpoint["best_validation_accuracy"]

            self.history = checkpoint["history"]

            self.logger.info(
                f"Resuming training from epoch {start_epoch}"
            )

        self.logger.info(f"\nDevice : {self.device}\n")

        for epoch in range(start_epoch, NUM_EPOCHS):

            epoch_start_time = time.perf_counter()

            train_loss, train_accuracy = self.engine.train_one_epoch(
                self.train_loader
            )

            validation_loss, validation_accuracy = self.engine.validate(
                self.validation_loader
            )

            self.scheduler.step(validation_loss)

            learning_rate = self.optimizer.param_groups[0]["lr"]

            epoch_time = (
                time.perf_counter()
                - epoch_start_time
            )

            self.history.update(
                train_loss=train_loss,
                validation_loss=validation_loss,
                train_accuracy=train_accuracy,
                validation_accuracy=validation_accuracy,
                learning_rate=learning_rate,
                epoch_time=epoch_time,
            )

            self.logger.info(

                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Val Acc: {validation_accuracy:.2f}% | "
                f"LR: {learning_rate:.6f} | "
                f"Time: {epoch_time:.2f}s"

            )

            self.checkpoint.save(

                epoch=epoch,

                best_validation_loss=min(
                    best_validation_loss,
                    validation_loss,
                ),

                best_validation_accuracy=max(
                    best_validation_accuracy,
                    validation_accuracy,
                ),

                history=self.history,

            )

            if validation_loss < best_validation_loss:

                best_validation_loss = validation_loss

                best_validation_accuracy = validation_accuracy

                self.checkpoint.save(

                    epoch=epoch,

                    best_validation_loss=best_validation_loss,

                    best_validation_accuracy=best_validation_accuracy,

                    history=self.history,

                )

            if self.early_stopping(validation_loss):

                self.logger.info(
                    "\nEarly stopping triggered.\n"
                )

                break

        self.logger.info("\nTraining completed.\n")

        self.logger.info(
            f"Best Validation Accuracy : "
            f"{best_validation_accuracy:.2f}%"
        )

        self.logger.info(
            f"Best Validation Loss : "
            f"{best_validation_loss:.4f}"
        )