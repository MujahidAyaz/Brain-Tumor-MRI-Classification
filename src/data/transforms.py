from torchvision import transforms

from configs.config import IMAGE_SIZE

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1840],
        std=[0.1895]
    ),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1840],
        std=[0.1895]
    ),
])