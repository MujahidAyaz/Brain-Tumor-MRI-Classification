# Import required libraries.
import torch

from sklearn.metrics import classification_report

from configs.config import (
    CLASS_NAMES,
)

from src.evaluation.metrics import calculate_metrics
from src.evaluation.report import save_evaluation_report
from src.evaluation.confusion_matrix import (
    save_confusion_matrix,
)


class Evaluator:
    """
    Evaluate the trained model on the test dataset.
    """

    # Initialize evaluator.
    def __init__(
        self,
        model,
        dataloader,
        device,
    ) -> None:

        self.model = model.to(device)

        self.dataloader = dataloader

        self.device = device

    # Evaluate the model.
    @torch.inference_mode()
    def evaluate(
        self,
    ) -> dict:

        self.model.eval()

        predictions : list[int] = []

        targets : list[int] = []

        for images, labels in self.dataloader:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(images)

            predicted_labels = outputs.argmax(
                dim=1,
            )

            predictions.extend(
                predicted_labels.cpu().tolist()
            )

            targets.extend(
                labels.cpu().tolist()
            )

        metrics = calculate_metrics(
            targets=targets,
            predictions=predictions,
        )

        
        

        save_evaluation_report(
            metrics,
        )

        save_confusion_matrix(
            targets,
            predictions,
        )

        return metrics