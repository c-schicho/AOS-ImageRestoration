import torch


def init_device(device) -> torch.device:

    if device == "cuda":
        print("Running on cuda")

    elif device == "cpu":
        print("Running on cpu")
        print("Disclaimer: Running on cpu is provided for debug and compatiblity reasons but shouldn't be used to train the model.")

    elif device == "mps":
        print("Running on mps")
        print("Disclaimer: Running on Apples GPU was tested and the script should be working, but is not guaranteed, please use cuda if you encounter mayor bugs.")

    else:
        ValueError(f"Unknonw Device: {device}")

    return torch.device(device)
