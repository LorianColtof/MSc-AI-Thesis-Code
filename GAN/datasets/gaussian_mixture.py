import os
from math import pi
from typing import List

import torch
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn

from configuration import Dataset
from datasets import AbstractBaseDataset


class GaussianMixtureDataset(AbstractBaseDataset):
    _source_samples_plot: torch.Tensor
    _num_plot_samples = 10000
    data_dimension = 2

    _real_samples: torch.Tensor

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
            (self._num_plot_samples, latent_dimension), device=device)

        self._real_samples = next(iter(self.dataloader))[0][:self._num_plot_samples] \
            .cpu().detach()

    def _create_plot(self):
        plt.figure()
        plt.plot(self._real_samples[:, 0], self._real_samples[:, 1],
                 'o', alpha=0.2, color='b')
        plt.xticks([])
        plt.yticks([])

    def save_generated_data(self, generator_network: torch.nn.Module,
                            images_path: str, filename: str) -> List[str]:
        with torch.no_grad():
            data_fake = generator_network(self._source_samples_plot)\
                .detach().cpu()

        img_path = os.path.join(images_path, f'{filename}.pdf')

        self._create_plot()
        seaborn.kdeplot(data_fake[:, 0], data_fake[:, 1], zorder=0,
                        n_levels=10, shade=True)
        plt.savefig(img_path)

        data_path = os.path.join(images_path, f'{filename}.pt')
        torch.save(data_fake.cpu(), data_path)

        return [img_path, data_path]

    def save_real_data(self, images_path: str, filename: str) -> List[str]:
        img_path = os.path.join(images_path, f'{filename}.pdf')

        self._create_plot()
        plt.savefig(img_path)

        data_path = os.path.join(images_path, f'{filename}.pt')
        torch.save(self._real_samples.cpu(), data_path)

        return [img_path, data_path]

