from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Dataset
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

NUM_CLASSES = len(CLASS_NAMES)

# Image
IMAGE_SIZE = (224, 224)
NUM_CHANNELS = 3

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

# Early Stopping
PATIENCE = 5
MIN_DELTA = 0.001

# Random Seed
RANDOM_SEED = 42