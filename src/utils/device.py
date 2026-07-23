# Import required libraries.
import torch


# Select the best available device.
def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        device_name = torch.cuda.get_device_name(0)

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(f"Device      : CUDA")
        print(f"GPU         : {device_name}")
        print(f"GPU Memory  : {total_memory:.2f} GB")

        return device

    if torch.backends.mps.is_available():

        print("Device      : Apple MPS")

        return torch.device("mps")

    print("Device      : CPU")

    return torch.device("cpu")