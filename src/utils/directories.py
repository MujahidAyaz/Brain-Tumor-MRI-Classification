# Import required libraries.
from pathlib import Path

from configs.config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    METRIC_DIR,
    MODEL_DIR,
    PLOT_DIR,
    PREDICTION_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_IMAGES_DIR,
)


# Create all required project directories.
def create_directories() -> None:
    """
    Create all project directories required during training,
    evaluation, and inference.
    """

    directories = (
        CHECKPOINT_DIR,
        LOG_DIR,
        METRIC_DIR,
        MODEL_DIR,
        PLOT_DIR,
        PREDICTION_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_IMAGES_DIR,
    )

    for directory in directories:

        Path(directory).mkdir(
            parents=True,
            exist_ok=True,
        )