# Import required libraries.
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    NUM_WORKERS,
)
from src.data.datasets import (
    get_test_dataset,
    get_train_dataset,
)


# Create the training DataLoader.
def get_train_loader(
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create and return the training DataLoader.

    Parameters
    ----------
    pin_memory : bool, optional
        Enable pinned memory for faster GPU data transfer.
    """

    train_dataset = get_train_dataset()

    return DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )


# Create the testing DataLoader.
def get_test_loader(
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create and return the testing DataLoader.

    Parameters
    ----------
    pin_memory : bool, optional
        Enable pinned memory for faster GPU data transfer.
    """

    test_dataset = get_test_dataset()

    return DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )