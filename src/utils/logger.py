# Import required libraries.
import logging

from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path

from configs.config import TRAINING_LOG_NAME


def setup_logger(
    log_dir: Path,
    log_level: int = logging.INFO,
) -> Logger:
    """
    Configure and return the project logger.

    Parameters
    ----------
    log_dir : Path
        Directory where log files will be stored.

    log_level : int, optional
        Logging level, by default logging.INFO.

    Returns
    -------
    Logger
        Configured project logger.
    """

    # Create the log directory if it does not exist.
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = log_dir / TRAINING_LOG_NAME

    # Create the project logger.
    logger = logging.getLogger("BrainTumorMRI")

    # Prevent duplicate log messages.
    logger.handlers.clear()

    logger.propagate = False

    logger.setLevel(log_level)

    # Define the log message format.
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure console logging.
    console_handler = logging.StreamHandler()

    console_handler.setLevel(log_level)

    console_handler.setFormatter(formatter)

    # Configure file logging.
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(log_level)

    file_handler.setFormatter(formatter)

    # Register handlers.
    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    return logger