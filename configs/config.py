# Import required libraries.
from pathlib import Path


# Project paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"


# Output paths.
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

MODEL_DIR = OUTPUT_DIR / "models"

LOG_DIR = OUTPUT_DIR / "logs"

PLOT_DIR = OUTPUT_DIR / "plots"

METRIC_DIR = OUTPUT_DIR / "metrics"

PREDICTION_DIR = OUTPUT_DIR / "predictions"


# Dataset configuration.
DATASET_NAME = "Brain Tumor MRI Dataset"

TRAIN_DIR = RAW_DATA_DIR / DATASET_NAME / "Training"

TEST_DIR = RAW_DATA_DIR / DATASET_NAME / "Testing"

CLASS_NAMES = (
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
)

NUM_CLASSES = len(CLASS_NAMES)


# Image configuration.
IMAGE_SIZE = (224, 224)

NUM_CHANNELS = 1

# Dataset normalization.
DATASET_MEAN = (0.1840,)
DATASET_STD = (0.1895,)


# Training configuration.
BATCH_SIZE = 32

NUM_EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

GRADIENT_CLIP = 1.0

USE_AMP = True


# Learning rate scheduler.
LR_FACTOR = 0.5

LR_PATIENCE = 2

MIN_LR = 1e-6


# DataLoader configuration.
NUM_WORKERS = 0

PIN_MEMORY = False


# Early stopping.
PATIENCE = 5

MIN_DELTA = 1e-3


# Reproducibility.
RANDOM_SEED = 42


# Saved file names.
BEST_MODEL_NAME = "best_model.pth"

LAST_CHECKPOINT_NAME = "last_checkpoint.pth"

TRAINING_LOG_NAME = "training.log"

HISTORY_CSV_NAME = "history.csv"

METRICS_JSON_NAME = "metrics.json"

# Evaluation reports.
CLASSIFICATION_REPORT_NAME = "classification_report.txt"

CONFUSION_MATRIX_CSV_NAME = "confusion_matrix.csv"

# Prediction outputs.
PREDICTIONS_CSV_NAME = "predictions.csv"

# Saved plots.
LOSS_CURVE_NAME = "loss_curve.png"

ACCURACY_CURVE_NAME = "accuracy_curve.png"

LEARNING_RATE_CURVE_NAME = "learning_rate_curve.png"

CONFUSION_MATRIX_PLOT_NAME = "confusion_matrix.png"