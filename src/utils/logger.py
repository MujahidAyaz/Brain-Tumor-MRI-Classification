# Import required libraries.
import logging

from logging import Logger
from pathlib import Path


def setup_logger(

    log_dir: Path,

    log_name: str = "training.log",

) -> Logger:

    # Create log directory if it does not exist.
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = log_dir / log_name

    # Create project logger.
    logger = logging.getLogger("BrainTumorMRI")

    # Avoid duplicate handlers.
    if logger.hasHandlers():

        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(message)s",

        datefmt="%Y-%m-%d %H:%M:%S",

    )

    # Log to terminal.
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    # Log to file.
    file_handler = logging.FileHandler(

        log_file,

        encoding="utf-8",

    )

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    return logger