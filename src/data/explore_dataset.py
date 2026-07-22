from pathlib import Path
from PIL import Image

from configs.config import RAW_DATA_DIR


dataset_path = RAW_DATA_DIR / "Brain Tumor MRI Dataset"

train_path = dataset_path / "Training"
test_path = dataset_path / "Testing"


def explore(split_path: Path):

    print(f"\n{'='*50}")
    print(split_path.name.upper())
    print(f"{'='*50}")

    total_images = 0

    for class_dir in sorted(split_path.iterdir()):

        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))

        print(f"{class_dir.name:<15}: {len(images)} images")

        total_images += len(images)

    print(f"\nTotal Images : {total_images}")


def sample_information():

    sample = next(train_path.rglob("*.*"))

    image = Image.open(sample)

    print("\nSample Image")
    print("------------------------")
    print("Path :", sample.name)
    print("Size :", image.size)
    print("Mode :", image.mode)


if __name__ == "__main__":

    explore(train_path)
    explore(test_path)

    sample_information()