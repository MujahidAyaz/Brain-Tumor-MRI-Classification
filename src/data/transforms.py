# Import required libraries.
from torchvision import transforms

from configs.config import (
    DATASET_MEAN,
    DATASET_STD,
    IMAGE_SIZE,
)


# Create training image transformations.
def get_train_transform() -> transforms.Compose:
    """
    Return the transformation pipeline used during training.
    """

    return transforms.Compose([

        transforms.Grayscale(num_output_channels=1),

        transforms.Resize(
            IMAGE_SIZE,
            antialias=True,
        ),

        transforms.RandomHorizontalFlip(
            p=0.5,
        ),

        transforms.RandomRotation(
            degrees=10,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=DATASET_MEAN,
            std=DATASET_STD,
        ),
    ])


# Create validation/testing transformations.
def get_test_transform() -> transforms.Compose:
    """
    Return the transformation pipeline used during validation and testing.
    """

    return transforms.Compose([

        transforms.Grayscale(num_output_channels=1),

        transforms.Resize(
            IMAGE_SIZE,
            antialias=True,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=DATASET_MEAN,
            std=DATASET_STD,
        ),
    ])


# Create inference transformations.
def get_predict_transform() -> transforms.Compose:
    """
    Return the transformation pipeline used during inference.
    """

    return get_test_transform()