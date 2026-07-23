# Import required libraries.
from typing import Any

import torch

from logging import Logger


def get_device(
    logger: Logger,
) -> tuple[torch.device, dict[str, Any]]:
    """
    Detect the best available device and log its information.

    Parameters
    ----------
    logger : Logger
        Configured project logger.

    Returns
    -------
    tuple
        Selected torch device and device information.
    """

    # Configure CPU information.
    device = torch.device("cpu")

    device_info = {
        "device": "CPU",
        "gpu_name": None,
        "gpu_memory": None,
        "gpu_count": 0,
        "torch_version": torch.__version__,
    }

    # Use CUDA if available.
    if torch.cuda.is_available():

        device = torch.device("cuda")

        device_info.update({
            "device": "CUDA",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory": round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2,
            ),
            "gpu_count": torch.cuda.device_count(),
        })

    # Otherwise use Apple MPS if available.
    elif torch.backends.mps.is_available():

        device = torch.device("mps")

        device_info["device"] = "Apple MPS"

    # Log device information.
    logger.info("Device Information")
    logger.info(f"Device          : {device_info['device']}")
    logger.info(f"PyTorch Version : {device_info['torch_version']}")

    if device_info["gpu_name"] is not None:

        logger.info(f"GPU             : {device_info['gpu_name']}")
        logger.info(f"GPU Memory      : {device_info['gpu_memory']} GB")
        logger.info(f"GPU Count       : {device_info['gpu_count']}")

    return device, device_info