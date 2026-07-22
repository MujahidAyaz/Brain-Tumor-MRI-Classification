from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

from src.data.dataset import (
    train_dataset,
    test_dataset,
)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)