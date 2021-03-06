import os
from typing import List

import torch
import torch.utils.data as utilsdata
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torchvision.utils import save_image

from configuration import Dataset
from datasets import AbstractBaseDataset


class Cifar10Dataset(AbstractBaseDataset):
    _source_samples_plot: torch.Tensor

    _sample_image_size = 10

    def __init__(self, dataset_config: Dataset, num_workers: int,
                 device: torch.device,
                 batch_size: int, latent_dimension: int,
                 drop_last: bool = False):
        self.data_dimension = 3 * 32 * 32

        self.dataloader = utilsdata.DataLoader(
            CIFAR10(os.path.join(dataset_config.directory, 'cifar10'),
                    train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])),
            batch_size=batch_size, shuffle=True,
            drop_last=drop_last, pin_memory=True, num_workers=num_workers)

        self._source_samples_plot = torch.randn(
            (self._sample_image_size ** 2, latent_dimension), device=device)

    def save_generated_data(self, generator_network: torch.nn.Module, images_path: str,
                            filename: str) -> List[str]:
        data_fake = generator_network(self._source_samples_plot)
        img_path = os.path.join(images_path, f'{filename}.png')
        save_image(data_fake.reshape(-1, 3, 32, 32),
                   img_path, nrow=self._sample_image_size, normalize=True)

        return [img_path]

    def save_real_data(self, images_path: str, filename: str) -> List[str]:
        data_real = next(iter(
            self.dataloader))[0][:self._sample_image_size ** 2]
        img_path = os.path.join(images_path, f'{filename}.png')
        save_image(data_real, img_path,
                   nrow=self._sample_image_size, normalize=True)

        return [img_path]
