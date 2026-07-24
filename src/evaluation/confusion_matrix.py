# Import required libraries.
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix

from configs.config import (
    CLASS_NAMES,
    METRIC_DIR,
    CONFUSION_MATRIX_CSV_NAME,
    CONFUSION_MATRIX_PLOT_NAME,
)


# Save the confusion matrix.
def save_confusion_matrix(
    targets: list[int],
    predictions: list[int],
) -> None:
    """
    Save the confusion matrix as both CSV and PNG.

    Parameters
    ----------
    targets : list[int]
        Ground-truth labels.

    predictions : list[int]
        Model predictions.
    """

    # Create the output directory.
    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Compute confusion matrix.
    matrix = confusion_matrix(
        targets,
        predictions,
    )

    # Convert to DataFrame.
    dataframe = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    # Save CSV.
    csv_path = (
        METRIC_DIR
        / CONFUSION_MATRIX_CSV_NAME
    )

    dataframe.to_csv(
        csv_path,
        index=True,
    )

    # Create figure.
    figure, axis = plt.subplots(
        figsize=(8, 6),
    )

    image = axis.imshow(
        matrix,
        cmap="Blues",
    )

    plt.colorbar(
        image,
    )

    axis.set_title(
        "Confusion Matrix",
    )

    axis.set_xlabel(
        "Predicted Label",
    )

    axis.set_ylabel(
        "True Label",
    )

    axis.set_xticks(
        range(len(CLASS_NAMES)),
    )

    axis.set_yticks(
        range(len(CLASS_NAMES)),
    )

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        CLASS_NAMES,
    )

    # Write values inside each cell.
    for row in range(matrix.shape[0]):

        for column in range(matrix.shape[1]):

            axis.text(

                column,

                row,

                matrix[row, column],

                ha="center",

                va="center",

                color="white"
                if matrix[row, column] > matrix.max() / 2
                else "black",

            )

    figure.tight_layout()

    plot_path = (
        METRIC_DIR
        / CONFUSION_MATRIX_PLOT_NAME
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()