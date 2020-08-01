from typing import List

import torch
from torchvision.transforms import transforms

from configuration import Dataset
from .mnist_base import AbstractMnistBase


class RotatedMnistDataset(AbstractMnistBase):
    def __init__(self, dataset_config: Dataset, device: torch.device,
                 num_workers: int, batch_size: int, latent_dimension: int):

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        super().__init__(['0', '90', '180', '270'], transform, 1,
                         dataset_config, num_workers, batch_size)
