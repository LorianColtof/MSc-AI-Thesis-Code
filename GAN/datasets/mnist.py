import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from configuration import Dataset
from datasets.base import AbstractBaseDataset


class MnistDataset(AbstractBaseDataset):
    _source_samples_plot: torch.Tensor

    _sample_image_size = 10

    def __init__(self, dataset_config: Dataset, device: torch.device,
                 batch_size: int, latent_dimension: int):
        self.data_dimension = 28 * 28

        self.dataloader = torch.utils.data.DataLoader(
            MNIST(os.path.join(dataset_config.directory, 'mnist'),
                  train=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.5,))])),
            batch_size=batch_size, shuffle=True,
            pin_memory=True)

        self._source_samples_plot = torch.randn(
            (self._sample_image_size ** 2, latent_dimension), device=device)

    def save_generated_data(self, generator_network: torch.nn.Module,
                            images_path: str,
                            steps: int, epochs: int) -> None:

        data_fake = generator_network(self._source_samples_plot)
        save_image(data_fake.reshape(-1, 1, 28, 28),
                   os.path.join(images_path, 'epoch_{}_step_{}.png'.format(
                       epochs, steps)),
                   nrow=self._sample_image_size, normalize=True)
