import torch


def mps_is_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_to_str(device):
    if isinstance(device, torch.device):
        return device.type
    return str(device)


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
