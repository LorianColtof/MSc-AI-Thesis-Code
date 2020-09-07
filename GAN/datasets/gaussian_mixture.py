import os
from math import pi

import torch
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from configuration import Dataset
from datasets import AbstractBaseDataset


class GaussianMixtureDataset(AbstractBaseDataset):
    _source_samples_plot: torch.Tensor
    data_dimension = 2

    def __init__(self, dataset_config: Dataset, num_workers: int,
                 device: torch.device, batch_size: int,
                 latent_dimension: int, drop_last: bool = False):
        num_samples = 4 * 10 ** 4
        num_mixtures = 8
        radius = 10

        means_list = []

        for i in range(num_mixtures):
            angle = torch.tensor([(2 * pi * i) / num_mixtures], device=device)
            x = radius * torch.cos(angle)
            y = radius * torch.sin(angle)

            means_list.append([x, y])

        means = torch.tensor(means_list, device=device)
        variances = torch.ones((num_mixtures, 2), device=device)

        mix = D.Categorical(torch.ones(num_mixtures, device=device))
        comp = D.Independent(D.Normal(means, variances), 1)
        mixture = D.MixtureSameFamily(mix, comp)
        samples = mixture.sample_n(num_samples)

        self.dataloader = DataLoader(
            TensorDataset(samples),
            batch_size=batch_size, shuffle=True, drop_last=drop_last)

        self._source_samples_plot = torch.randn(
            (10**3, latent_dimension), device=device)

    def save_generated_data(self, generator_network: torch.nn.Module,
                            images_path: str, filename: str) -> str:
        with torch.no_grad():
            data_fake = generator_network(self._source_samples_plot)\
                .detach().cpu()

        img_path = os.path.join(images_path, f'{filename}.png')

        plt.figure()
        plt.plot(data_fake[:, 0], data_fake[:, 1], '.', alpha=0.1, color='b')
        plt.savefig(img_path)

        return img_path

