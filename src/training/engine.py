# Import required libraries.
import torch

from tqdm import tqdm


class Engine:

    # Initialize training engine.
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
    ):

        self.model = model

        self.criterion = criterion
        
        self.optimizer = optimizer

        self.device = device

    # Train the model for one epoch.
    def train_one_epoch(
        self,
        dataloader,
    ):

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

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()

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

    # Validate the model.
    @torch.inference_mode()
    def validate(
        self,
        dataloader,
    ):

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