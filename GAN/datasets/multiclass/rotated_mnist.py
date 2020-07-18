import os
from typing import List, Dict

import torch
from torch import Tensor
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets

from .base import AbstractBaseMulticlassDataset
from configuration import Dataset


class RotatedMnistDataset(AbstractBaseMulticlassDataset):
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

    def save_generated_data(self, source_data: Tensor,
                            generated_data: Dict[str, Tensor],
                            images_path: str, filename: str) -> str:
        num_samples = source_data.size(0)

        img_list = []
        for i in range(num_samples):
            img_src = source_data[i, :, :, :]
            img_list.append(img_src.cpu())
            for _class in sorted(generated_data.keys()):
               img_tgt = generated_data[_class][i]
               img_list.append(img_tgt.cpu())

        path = os.path.join(images_path, filename + '.png')
        save_image(img_list, path, nrow=len(self.classes),
                   padding=0, normalize=True)

        return path

    @property
    def target_classes(self) -> List[str]:
        # Make sure they are sorted nicely
        return ['90', '180', '270']
