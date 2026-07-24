# Import required libraries.
import torch

from configs.config import (
    BEST_MODEL_NAME,
    MODEL_DIR,
    LOG_DIR,
    RANDOM_SEED,
)

from src.data.dataloader import get_test_loader
from src.evaluation.evaluator import Evaluator
from src.models.cnn import BrainTumorCNN

from src.utils.device import get_device
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


# Run the complete evaluation pipeline.
def main() -> None:

    # Ensure reproducible results.
    set_seed(RANDOM_SEED)

    # Configure the logger.
    logger = setup_logger(LOG_DIR)

    # Select the best available device.
    device, _ = get_device(logger)

    # Create the model.
    model = BrainTumorCNN()

    # Load trained weights.
    model_path = MODEL_DIR / BEST_MODEL_NAME
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found: {model_path}"
        )
    
    model.load_state_dict(

        torch.load(

            model_path,

            map_location=device,

            weights_only=True,

        )

    )

    # Create the testing DataLoader.
    test_loader = get_test_loader()

    # Create evaluator.
    evaluator = Evaluator(

        model=model,

        dataloader=test_loader,

        device=device,

    )

    # Run evaluation.
    metrics = evaluator.evaluate()

    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation Results")
    logger.info("=" * 70)

    logger.info(
        f"Accuracy  : {metrics['accuracy']:.2f}%"
    )

    logger.info(
        f"Precision : {metrics['precision']:.2f}%"
    )

    logger.info(
        f"Recall    : {metrics['recall']:.2f}%"
    )

    logger.info(
        f"F1 Score  : {metrics['f1_score']:.2f}%"
    )

    logger.info("=" * 70)


# Execute the program.
if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        raise RuntimeError(

            f"Evaluation failed: {error}"

        ) from error