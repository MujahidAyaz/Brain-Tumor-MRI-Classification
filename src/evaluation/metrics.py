# Import required libraries.
from typing import Any


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from configs.config import CLASS_NAMES


# Calculate evaluation metrics.
EvaluationMetrics = dict[str, Any]

def calculate_metrics(
    targets: list[int],
    predictions: list[int],
) -> EvaluationMetrics:
    """
    Calculate all evaluation metrics.

    Parameters
    ----------
    targets : list[int]
        Ground-truth labels.

    predictions : list[int]
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """

    accuracy = accuracy_score(
        targets,
        predictions,
    ) * 100

    precision = precision_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    ) * 100

    recall = recall_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    ) * 100

    f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    ) * 100

    report = classification_report(
        targets,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "classification_report": report,

    }