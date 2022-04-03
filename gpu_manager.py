import torch
from torch.utils.data import DataLoader


def manage_gpu(train_dl):
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    return train_dl, device


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")  # FIXME: Might be getting integrated graphics (do "cuda:1")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader(DataLoader):
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        super().__init__(dl)
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
