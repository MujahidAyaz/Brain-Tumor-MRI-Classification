# Import required libraries.
import json


from configs.config import (
    METRIC_DIR,
    METRICS_JSON_NAME,
    CLASSIFICATION_REPORT_NAME,
)


# Save evaluation metrics.
def save_evaluation_report(
    metrics: dict,
) -> None:
    """
    Save evaluation metrics to disk.

    Parameters
    ----------
    metrics : dict
        Dictionary returned by calculate_metrics().
    """

    # Create the metrics directory.
    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save metrics as JSON.
    metrics_json_path = (
        METRIC_DIR
        / METRICS_JSON_NAME
    )

    with open(
        metrics_json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Save classification report.
    report_path = (
        METRIC_DIR
        / CLASSIFICATION_REPORT_NAME
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Brain Tumor MRI Classification Report\n"
        )

        file.write("=" * 60)

        file.write("\n\n")

        file.write(
            f"Accuracy : {metrics['accuracy']:.2f}%\n"
        )

        file.write(
            f"Precision : {metrics['precision']:.2f}%\n"
        )

        file.write(
            f"Recall : {metrics['recall']:.2f}%\n"
        )

        file.write(
            f"F1 Score : {metrics['f1_score']:.2f}%\n"
        )

        file.write("\n")

        file.write("=" * 60)

        file.write("\n\n")

        for class_name, values in metrics[
            "classification_report"
        ].items():

            file.write(f"{class_name}\n")

            file.write("-" * 40)

            file.write("\n")

            if isinstance(
                values,
                dict,
            ):

                for metric, value in values.items():

                    if isinstance(value, float):

                        file.write(
                            f"{metric:<15}: {value:.4f}\n"
                        )

                    else:

                        file.write(
                            f"{metric:<15}: {value}\n"
                        )

            else:

                file.write(
                    f"{values}\n"
                )

            file.write("\n")