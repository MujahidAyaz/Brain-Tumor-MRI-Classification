# Import required libraries.
from configs.config import (
    LOG_DIR,
    RANDOM_SEED,
)

from src.data.dataloader import (
    get_test_loader,
    get_train_loader,
)

from src.models.cnn import BrainTumorCNN

from src.training.trainer import Trainer

from src.utils.device import get_device
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


# Run the complete training pipeline.
def main() -> None:
    """
    Initialize and execute the complete training workflow.
    """

    # Set random seed for reproducibility.
    set_seed(RANDOM_SEED)

    # Configure the project logger.
    logger = setup_logger(LOG_DIR)

    # Select the best available device.
    device, _ = get_device(logger)

    # Create the training DataLoader.
    train_loader = get_train_loader()

    # Create the validation DataLoader.
    validation_loader = get_test_loader()

    # Build the CNN model.
    model = BrainTumorCNN()

    # Initialize the trainer.
    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        validation_loader=validation_loader,

        device=device,

        log_dir=LOG_DIR,

    )

    # Start model training.
    trainer.train()


# Execute the training script.
if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nTraining interrupted by user.")

    except Exception as error:

        raise RuntimeError(
            f"Training failed: {error}"
        ) from error