import io
import os
import pickle
from math import pi
from typing import Dict, List, Any, Tuple

import seaborn
import torch
from matplotlib import pyplot as plt
from torch import Tensor
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
from torch.serialization import DEFAULT_PROTOCOL

from configuration import Dataset
from datasets.multiclass.base import AbstractBaseMulticlassDataset


class ToyMixtureDataset(AbstractBaseMulticlassDataset):
    data_dimension = 2
    source_class = 'center'
    _num_plot_samples = 10000

    _source_samples: torch.Tensor
    _pickle_data: Dict[str, bytes]

    _plot_colors = ['red', 'blue', 'orange', 'purple', 'brown', 'tan']

    def __init__(self, dataset_config: Dataset, device: torch.device,
                 num_workers: int, batch_size: int, latent_dimension: int):
        num_mixtures = 6
        num_samples = 4 * 10 ** 4
        radius = 1.5

        def create_dataloader(x: float, y: float) -> DataLoader:
            mean = torch.tensor([x, y], device=device)
            scale = torch.Tensor([0.2], device=device)
            dist = D.Normal(mean, scale)
            samples = dist.sample((num_samples, ))

            return DataLoader(
                TensorDataset(samples),
                batch_size=batch_size, shuffle=True, drop_last=True)

        def sample_dataloader(dataloader: DataLoader) -> Tuple[torch.Tensor, bytes]:
            samples = next(iter(dataloader))[0][:self._num_plot_samples].cpu().detach()
            buffer = io.BytesIO()
            torch.save(samples, buffer)
            samples_data = buffer.getvalue()
            buffer.close()

            return samples, samples_data

        self.source_dataloader = create_dataloader(0.0, 0.0)
        self._source_samples, source_samples_data =\
            sample_dataloader(self.source_dataloader)

        self._pickle_data = {
            'source': source_samples_data
        }

        self.target_dataloaders = {}
        self._target_samples = {}

        for i in range(num_mixtures):
            angle = torch.tensor([(2 * pi * i) / num_mixtures], device=device)
            x = radius * torch.cos(angle)
            y = radius * torch.sin(angle)

            _class = str(i)
            self.target_dataloaders[_class] = create_dataloader(x, y)

            samples, samples_data = sample_dataloader(self.target_dataloaders[_class])
            self._target_samples[_class] = samples
            self._pickle_data[f'{_class}_real'] = samples_data

    def save_generated_data(self, source_data: Tensor,
                            generated_data: Dict[str, Tensor], images_path: str,
                            filename: str) -> List[str]:
        plt.figure()
        plt.xticks([])
        plt.yticks([])

        plt.plot(self._source_samples[:, 0], self._source_samples[:, 1],
                 'o', alpha=0.2, color='g')

        for i, (_class, data) in enumerate(generated_data.items()):
            target_samples = \
                next(iter(self.target_dataloaders[_class]))[0][:self._num_plot_samples]\
                .cpu().detach()
            plt.plot(target_samples[:, 0], target_samples[:, 1],
                     'o', alpha=0.2, color=self._plot_colors[i])

            data_cpu = data.cpu().detach()
            seaborn.kdeplot(x=data_cpu[:, 0], y=data_cpu[:, 1], zorder=0,
                            n_levels=5, shade=True, color=self._plot_colors[i])

            buffer = io.BytesIO()
            torch.save(data_cpu, buffer)
            self._pickle_data[f'{_class}_generated'] = buffer.getvalue()
            buffer.close()

        img_path = os.path.join(images_path, f'{filename}.pdf')
        plt.savefig(img_path)

        data_path = os.path.join(images_path, f'{filename}.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(self._pickle_data, f, protocol=DEFAULT_PROTOCOL)

        return [img_path, data_path]

