# Import required libraries.
from torchvision.datasets import ImageFolder

from configs.config import RAW_DATA_DIR
from src.data.transforms import (
    get_test_transform,
    get_train_transform,
)


# Define dataset locations.
from configs.config import TRAIN_DIR, TEST_DIR


# Create the training dataset.
def get_train_dataset() -> ImageFolder:
    """
    Create and return the training dataset.
    """

    return ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transform(),
    )


# Create the testing dataset.
def get_test_dataset() -> ImageFolder:
    """
    Create and return the testing dataset.
    """

    return ImageFolder(
        root=TEST_DIR,
        transform=get_test_transform(),
    )