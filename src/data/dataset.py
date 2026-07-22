from torchvision.datasets import ImageFolder

from configs.config import RAW_DATA_DIR

from src.data.transforms import (
    train_transform,
    test_transform,
)

TRAIN_DIR = RAW_DATA_DIR / "Brain Tumor MRI Dataset" / "Training"

TEST_DIR = RAW_DATA_DIR / "Brain Tumor MRI Dataset" / "Testing"

train_dataset = ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform,
)

test_dataset = ImageFolder(
    root=TEST_DIR,
    transform=test_transform,
)