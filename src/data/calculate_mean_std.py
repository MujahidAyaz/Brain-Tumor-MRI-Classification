import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from configs.config import IMAGE_SIZE
from configs.config import NUM_WORKERS

from configs.config import (
    RAW_DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

train_dir = RAW_DATA_DIR / "Brain Tumor MRI Dataset" / "Training"


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.ToTensor(),
])


dataset = datasets.ImageFolder(
    root=train_dir,
    transform=transform,
)



loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


def calculate_mean_std(data_loader):

    mean = 0.0
    std = 0.0
    total_batches = 0

    for images, _ in data_loader:

        mean += images.mean(dim=[0, 2, 3])
        std += images.std(dim=[0, 2, 3])

        total_batches += 1

    mean /= total_batches
    std /= total_batches

    return mean, std


if __name__ == "__main__":
    print(f"NUM_WORKERS = {NUM_WORKERS}")
    print(dataset[0][0].shape)
    print(dataset.classes)

    mean, std = calculate_mean_std(loader)

    print(f"Mean : {mean}")
    print(f"Std  : {std}")