# Import required libraries.
import time

import torch

from torch import nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    LR_FACTOR,
    LR_PATIENCE,
    MIN_LR,
    NUM_EPOCHS,
    PATIENCE,
    MIN_DELTA,
    CHECKPOINT_DIR,
    MODEL_DIR,
)

from src.training.engine import Engine
from src.training.history import TrainingHistory
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import CheckpointManager

from src.utils.logger import setup_logger


class Trainer:
    """
    Manage the complete model training pipeline.
    """

    # Initialize the trainer.
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        validation_loader,
        device: torch.device,
        log_dir,
    ) -> None:

        self.device = device

        self.model = model.to(device)

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        # Configure project logger.
        self.logger = setup_logger(log_dir)

        # Configure loss function.
        self.criterion = nn.CrossEntropyLoss()

        # Configure optimizer.
        self.optimizer = AdamW(

            params=self.model.parameters(),

            lr=LEARNING_RATE,

            weight_decay=WEIGHT_DECAY,

        )

        # Configure learning-rate scheduler.
        self.scheduler = ReduceLROnPlateau(

            optimizer=self.optimizer,

            mode="min",

            factor=LR_FACTOR,

            patience=LR_PATIENCE,

            min_lr=MIN_LR,

        )

        # Configure automatic mixed precision.
        self.scaler = GradScaler(

            "cuda",

            enabled=device.type == "cuda",

        )

        # Configure training engine.
        self.engine = Engine(

            model=self.model,

            criterion=self.criterion,

            optimizer=self.optimizer,

            device=self.device,

            scaler=self.scaler,

        )

        # Configure training history.
        self.history = TrainingHistory()

        # Configure early stopping.
        self.early_stopping = EarlyStopping(

            patience=PATIENCE,

            min_delta=MIN_DELTA,

        )

        # Configure checkpoint manager.
        self.checkpoint = CheckpointManager(

            model=self.model,

            optimizer=self.optimizer,

            scheduler=self.scheduler,

            checkpoint_dir=CHECKPOINT_DIR,

            model_dir=MODEL_DIR,

        )

            # Train the model.
    def train(self) -> TrainingHistory:

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Starting Brain Tumor MRI Classification Training")
        self.logger.info("=" * 70)

        start_epoch = 0

        best_validation_loss = float("inf")

        best_validation_accuracy = 0.0

        total_training_start = time.perf_counter()

        # Resume training if a checkpoint exists.
        checkpoint = self.checkpoint.load_checkpoint(
            self.device,
        )

        if checkpoint is not None:

            start_epoch = checkpoint["epoch"] + 1

            best_validation_loss = checkpoint[
                "best_validation_loss"
            ]

            best_validation_accuracy = checkpoint[
                "best_validation_accuracy"
            ]

            self.history = TrainingHistory.from_dict(
                checkpoint["history"]
            )

            self.logger.info(
                f"Resumed from Epoch {start_epoch}"
            )

        else:

            self.logger.info(
                "No checkpoint found. Starting fresh training."
            )

        self.logger.info(f"Device          : {self.device}")

        self.logger.info(
            f"Training Epochs : {NUM_EPOCHS}"
        )

        self.logger.info("-" * 70)

        for epoch in range(
            start_epoch,
            NUM_EPOCHS,
        ):

            self.logger.info("")

            self.logger.info(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
            )

            epoch_start = time.perf_counter()

            train_loss, train_accuracy = (
                self.engine.train_one_epoch(
                    self.train_loader,
                )
            )

            validation_loss, validation_accuracy = (
                self.engine.validate_one_epoch(
                    self.validation_loader,
                )
            )

            self.scheduler.step(
                validation_loss,
            )

            learning_rate = (
                self.optimizer.param_groups[0]["lr"]
            )

            epoch_time = (
                time.perf_counter()
                - epoch_start
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
                f"Train Loss      : {train_loss:.4f}"
            )

            self.logger.info(
                f"Train Accuracy  : {train_accuracy:.2f}%"
            )

            self.logger.info(
                f"Validation Loss : {validation_loss:.4f}"
            )

            self.logger.info(
                f"Validation Acc  : {validation_accuracy:.2f}%"
            )

            self.logger.info(
                f"Learning Rate   : {learning_rate:.6f}"
            )

            self.logger.info(
                f"Epoch Time      : {epoch_time:.2f} sec"
            )


                        
            

            # Save the best model if validation loss improves.
            if validation_loss < best_validation_loss:

                best_validation_loss = validation_loss

                best_validation_accuracy = validation_accuracy

                self.checkpoint.save_best_model()

                self.logger.info(
                    "Best model updated."
                )

            # Save the latest checkpoint after every epoch.
            self.checkpoint.save_checkpoint(

                epoch=epoch,

                best_validation_loss=best_validation_loss,

                best_validation_accuracy=best_validation_accuracy,

                history=self.history,

            )

            # Check whether training should stop early.
            if self.early_stopping(
                validation_loss,
            ):

                self.logger.info("")
                self.logger.info(
                    "Early stopping triggered."
                )

                break

            self.logger.info("-" * 70)

        total_training_time = (
            time.perf_counter()
            - total_training_start
        )

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Training Completed")
        self.logger.info("=" * 70)

        self.logger.info(
            f"Best Validation Accuracy : "
            f"{best_validation_accuracy:.2f}%"
        )

        self.logger.info(
            f"Best Validation Loss     : "
            f"{best_validation_loss:.4f}"
        )

        self.logger.info(
            f"Total Training Time      : "
            f"{total_training_time:.2f} sec"
        )

        self.logger.info("=" * 70)

        return self.history