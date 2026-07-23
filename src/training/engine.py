# Import required libraries.
import torch

from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import GRADIENT_CLIP


class Engine:
    """
    Handle one epoch of training and validation.
    """

    # Initialize the training engine.
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler: GradScaler,
    ) -> None:

        self.model = model

        self.criterion = criterion

        self.optimizer = optimizer

        self.device = device

        self.scaler = scaler

        self.use_amp = device.type == "cuda"

    # Train the model for one epoch.
    def train_one_epoch(
        self,
        dataloader: DataLoader,
    ) -> tuple[float, float]:

        self.model.train()

        running_loss = 0.0

        correct_predictions = 0

        total_samples = 0

        progress_bar = tqdm(
            dataloader,
            desc="Training",
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in progress_bar:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(
                set_to_none=True,
            )

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels,
                )

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=GRADIENT_CLIP,
            )

            self.scaler.step(
                self.optimizer,
            )

            self.scaler.update()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            current_loss = (
                running_loss
                / (progress_bar.n + 1)
            )

            current_accuracy = (
                100
                * correct_predictions
                / total_samples
            )

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            progress_bar.set_postfix(

                Loss=f"{current_loss:.4f}",

                Acc=f"{current_accuracy:.2f}%",

                LR=f"{current_lr:.6f}",

            )

        epoch_loss = (
            running_loss
            / len(dataloader)
        )

        epoch_accuracy = (
            100
            * correct_predictions
            / total_samples
        )

        return epoch_loss, epoch_accuracy

    # Validate the model for one epoch.
    @torch.inference_mode()
    def validate_one_epoch(
        self,
        dataloader: DataLoader,
    ) -> tuple[float, float]:

        self.model.eval()

        running_loss = 0.0

        correct_predictions = 0

        total_samples = 0

        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in progress_bar:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels,
                )

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            current_loss = (
                running_loss
                / (progress_bar.n + 1)
            )

            current_accuracy = (
                100
                * correct_predictions
                / total_samples
            )

            progress_bar.set_postfix(

                Loss=f"{current_loss:.4f}",

                Acc=f"{current_accuracy:.2f}%",

            )

        epoch_loss = (
            running_loss
            / len(dataloader)
        )

        epoch_accuracy = (
            100
            * correct_predictions
            / total_samples
        )

        return epoch_loss, epoch_accuracy