# Import required libraries.
from configs.config import (
    RANDOM_SEED,
    LOG_DIR,
)

from src.data.dataloader import (
    get_train_loader,
    get_test_loader,
)

from src.models.cnn import BrainTumorCNN

from src.training.trainer import Trainer

from src.utils.device import get_device
from src.utils.seed import set_seed


# Train the complete model.
def main() -> None:
    """
    Initialize the complete training pipeline.
    """

    # Set random seed for reproducibility.
    set_seed(RANDOM_SEED)

    # Select the best available device.
    device = get_device()

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


# Run the training pipeline.
if __name__ == "__main__":

    main()