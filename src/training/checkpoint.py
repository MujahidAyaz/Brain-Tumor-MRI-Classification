# Import required libraries.
import torch


class CheckpointManager:

    # Initialize checkpoint manager.
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        checkpoint_path,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_path = checkpoint_path

    # Save complete training checkpoint.
    def save(
        self,
        epoch,
        best_validation_loss,
        best_validation_accuracy,
        history,
    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),

            "scheduler_state_dict": self.scheduler.state_dict(),

            "best_validation_loss": best_validation_loss,

            "best_validation_accuracy": best_validation_accuracy,

            "history": history,
        }

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )

    # Load checkpoint if available.
    def load(self):

        if not self.checkpoint_path.exists():

            return None

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
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