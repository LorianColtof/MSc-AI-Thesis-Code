import torch
from torchvision.transforms import transforms

from configuration import Dataset
from .mnist_base import AbstractMnistBase


class ColoredMnistDataset(AbstractMnistBase):
    def __init__(self, dataset_config: Dataset, device: torch.device,
                 num_workers: int, batch_size: int, latent_dimension: int):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        super().__init__(['white', 'red', 'green', 'blue'], transform, 3,
                         dataset_config, num_workers, batch_size)
