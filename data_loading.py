import torch
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# TODO: Cleanup and understand code
def load_data(batch_size: int, stats):
    torch.multiprocessing.freeze_support()
    data_dir = "Data/raw images/Abstract_gallery"
    image_size = 64
    transform = {
        "train": tt.Compose([
            tt.Resize(image_size),
            tt.CenterCrop(image_size),
            tt.ToTensor(),
            tt.Normalize(*stats)])
    }

    train_ds = ImageFolder(data_dir, transform=transform["train"])
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_dl


def denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, stats, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax], stats), nrow=8).permute(1, 2, 0))


def show_batch(dl, stats, nmax=64):
    for images, _ in dl:
        show_images(images, stats, nmax)
        break
    plt.show()
