# Import required libraries.
import os
import random

import numpy as np
import torch

from logging import Logger


# Set random seed for reproducibility.
def set_seed(
    seed: int,
    logger: Logger | None = None,
) -> None:
    """
    Configure random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed value.

    logger : Logger | None, optional
        Project logger used to record the selected seed.
    """

    # Configure Python hash seed.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Configure Python random module.
    random.seed(seed)

    # Configure NumPy.
    np.random.seed(seed)

    # Configure PyTorch CPU.
    torch.manual_seed(seed)

    # Configure PyTorch CUDA.
    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms when supported.
    try:

        torch.use_deterministic_algorithms(True)

    except RuntimeError:

        pass

    # Log the configured seed.
    if logger is not None:

        logger.info(f"Random Seed : {seed}")