import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from configuration import Configuration
from datasets.base import AbstractBaseDataset


class MnistDataset(AbstractBaseDataset):
    _source_samples_plot: torch.Tensor

    def __init__(self, config: Configuration):
        self.data_dimension = 28 * 28

        self.dataloader = torch.utils.data.DataLoader(
            MNIST(os.path.join(config.dataset.directory, 'mnist'),
                  train=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.5,))])),
            batch_size=config.train.batch_size, shuffle=True,
            pin_memory=True)

        self._source_samples_plot = torch.randn(
            (5 * 5, config.train.latent_dimension),
            device=config.runtime_options['device'])

    def save_generated_data(self, generator_network: torch.nn.Module,
                            images_path: str,
                            steps: int, epochs: int) -> None:

        data_fake = generator_network(self._source_samples_plot)
        save_image(data_fake.reshape(-1, 1, 28, 28),
                   os.path.join(images_path, 'epoch_{}_step_{}.png'.format(
                       epochs, steps)), nrow=5, normalize=True)
