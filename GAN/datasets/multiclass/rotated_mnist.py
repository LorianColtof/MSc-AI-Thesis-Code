import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets

from .base import AbstractBaseMulticlassDataset
from configuration import Dataset


class RotatedMnistDataset(AbstractBaseMulticlassDataset):
    _source_samples_plot: torch.Tensor
    _sample_image_size = 10

    def __init__(self, dataset_config: Dataset, device: torch.device,
                 num_workers: int, batch_size: int, latent_dimension: int,
                 drop_last: bool = False):
        self.data_dimension = 28 * 28

        all_classes = {'0', '90', '180', '270'}
        if not dataset_config.source_class:
            raise Exception("dataset.source_class is required "
                            "for multi-class datasets")

        self.source_class = dataset_config.source_class
        if self.source_class not in all_classes:
            raise Exception(f"Invalid source class: {self.source_class}. "
                            f"Allowed classes: {','.join(all_classes)}")

        self.target_dataloaders = {}
        for _class in all_classes:
            dataloader = DataLoader(
                datasets.ImageFolder(
                    os.path.join(dataset_config.directory, _class),
                    transform=transforms.Compose([
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])),
                batch_size=batch_size, shuffle=True, pin_memory=True,
                drop_last=drop_last, num_workers=num_workers)

            if _class == self.source_class:
                self.source_dataloader = dataloader
            else:
                self.target_dataloaders[_class] = dataloader