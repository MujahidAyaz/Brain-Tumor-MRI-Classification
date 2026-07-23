from pathlib import Path


# Project Paths


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"



# Output Paths

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

MODEL_DIR = OUTPUT_DIR / "models"

LOG_DIR = OUTPUT_DIR / "logs"

PLOT_DIR = OUTPUT_DIR / "plots"

METRIC_DIR = OUTPUT_DIR / "metrics"

PREDICTION_DIR = OUTPUT_DIR / "predictions"



# Dataset


CLASS_NAMES = (
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
)

NUM_CLASSES = len(CLASS_NAMES)

# Dataset
DATASET_NAME = "Brain Tumor MRI Dataset"

TRAIN_DIR = RAW_DATA_DIR / DATASET_NAME / "Training"

TEST_DIR = RAW_DATA_DIR / DATASET_NAME / "Testing"

# Image


IMAGE_SIZE = (224, 224)

NUM_CHANNELS = 1



# Training


BATCH_SIZE = 32

NUM_EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

GRADIENT_CLIP = 1.0

USE_AMP = True



# Learning Rate Scheduler


LR_FACTOR = 0.5

LR_PATIENCE = 2

MIN_LR = 1e-6


# DataLoader


NUM_WORKERS = 0



# Early Stopping


PATIENCE = 5

MIN_DELTA = 1e-3



# Reproducibility


RANDOM_SEED = 42



# Saved Files


BEST_MODEL_NAME = "best_model.pth"

LAST_CHECKPOINT_NAME = "last_checkpoint.pth"

TRAINING_LOG_NAME = "training.log"

HISTORY_CSV_NAME = "history.csv"

METRICS_JSON_NAME = "metrics.json"

# Gradient clipping.
GRADIENT_CLIP = 1.0

# Dataset normalization values.
DATASET_MEAN = (0.1840,)

DATASET_STD = (0.1895,)




