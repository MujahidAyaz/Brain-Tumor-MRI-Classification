# Import required libraries.
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from configs.config import (
    BEST_MODEL_NAME,
    LAST_CHECKPOINT_NAME,
)


class CheckpointManager:
    """
    Manage saving and loading of training checkpoints.
    """

    # Initialize the checkpoint manager.
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        checkpoint_dir: Path,
        model_dir: Path,
    ) -> None:

        self.model = model

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.checkpoint_path = (
            checkpoint_dir / LAST_CHECKPOINT_NAME
        )

        self.best_model_path = (
            model_dir / BEST_MODEL_NAME
        )

    # Save the complete training checkpoint.
    def save_checkpoint(
        self,
        epoch: int,
        best_validation_loss: float,
        best_validation_accuracy: float,
        history: Any,
    ) -> None:

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "best_validation_loss":
                best_validation_loss,

            "best_validation_accuracy":
                best_validation_accuracy,

            "history":
                history.to_dict(),
        }

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )

    # Save only the best model weights.
    def save_best_model(self) -> None:

        torch.save(
            self.model.state_dict(),
            self.best_model_path,
        )

    # Load the latest checkpoint.
    def load_checkpoint(
        self,
        device: torch.device,
    ) -> dict[str, Any] | None:

        if not self.checkpoint_path.exists():

            return None

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        return checkpoint