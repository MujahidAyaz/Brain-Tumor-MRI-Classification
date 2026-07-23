# Import required libraries.
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from configs.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    MIN_DELTA,
    MODEL_DIR,
)


class Trainer:

    # Initialize the trainer.
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device,
    ):

        self.model = model.to(device)

        self.train_loader = train_loader
        self.test_loader = test_loader

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

        # Store training history.
        self.train_losses = []
        self.validation_losses = []

        self.train_accuracies = []
        self.validation_accuracies = []

        # Configure early stopping.
        self.best_validation_loss = float("inf")
        self.best_validation_accuracy = 0.0
        self.early_stop_counter = 0

        # Create model directory.
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train the model for one epoch.
    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        )

        for images, labels in progress_bar:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()

            total_samples += labels.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct_predictions / total_samples:.2f}%"
            )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct_predictions / total_samples

        return epoch_loss, epoch_accuracy

    # Validate the model.
    @torch.inference_mode()
    def validate(self):

        self.model.eval()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(
            self.test_loader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress_bar:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()

            total_samples += labels.size(0)

        epoch_loss = running_loss / len(self.test_loader)
        epoch_accuracy = 100 * correct_predictions / total_samples

        return epoch_loss, epoch_accuracy

    # Save the best performing model.
    def save_best_model(self):

        torch.save(
            self.model.state_dict(),
            MODEL_DIR / "best_model.pth",
        )

    # Train the complete model.
    def train(self):

        print(f"\nTraining on {self.device}\n")

        for epoch in range(NUM_EPOCHS):

            train_loss, train_accuracy = self.train_one_epoch()

            validation_loss, validation_accuracy = self.validate()

            self.scheduler.step(validation_loss)

            self.train_losses.append(train_loss)
            self.validation_losses.append(validation_loss)

            self.train_accuracies.append(train_accuracy)
            self.validation_accuracies.append(validation_accuracy)

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Val Acc: {validation_accuracy:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            if validation_loss < (self.best_validation_loss - MIN_DELTA):

                self.best_validation_loss = validation_loss
                self.best_validation_accuracy = validation_accuracy

                self.early_stop_counter = 0

                self.save_best_model()

            else:

                self.early_stop_counter += 1

            if self.early_stop_counter >= PATIENCE:

                print("\nEarly stopping triggered.\n")
                break

        print("\nTraining completed successfully.")