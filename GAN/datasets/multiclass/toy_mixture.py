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
    _num_plot_samples = 128

    _source_samples: torch.Tensor
    _pickle_data: Dict[str, bytes]

    _plot_colors = ['red', 'blue', 'orange', 'purple', 'brown', 'tan']

    def __init__(self, dataset_config: Dataset, device: torch.device,
                 num_workers: int, batch_size: int, latent_dimension: int):
        num_mixtures = 6
        num_samples = 4 * 10 ** 4
        radius = 1.5

        if dataset_config.target_type:
            target_type = dataset_config.target_type
            allowed_target_types = {"gaussian", "uniform"}
            if target_type not in allowed_target_types:
                raise Exception(
                    f"target_type must be one of: {','.join(allowed_target_types)}")
        else:
            target_type = "gaussian"

        def create_dataloader(x: float, y: float, type: str) -> DataLoader:
            mean = torch.tensor([x, y], device=device)

            if type == "gaussian":
                scale = torch.tensor([0.2], device=device)
                dist = D.Normal(mean, scale)
            else:
                low = mean - 0.2
                high = mean + 0.2
                dist = D.Uniform(low, high)

            samples = dist.sample((num_samples, ))

            return DataLoader(
                TensorDataset(samples),
                batch_size=batch_size, shuffle=True, drop_last=True)

        def sample_dataloader(dataloader: DataLoader) -> Tuple[torch.Tensor, bytes]:
            sample_list = []
            num_samples = 0

            while num_samples < self._num_plot_samples:
                samples_batch = next(iter(dataloader))[0].cpu().detach()
                sample_list.append(samples_batch)
                num_samples += samples_batch.size(0)

            samples = torch.cat(sample_list, dim=0)[:self._num_plot_samples]
            buffer = io.BytesIO()
            torch.save(samples, buffer)
            samples_data = buffer.getvalue()
            buffer.close()

            return samples, samples_data

        self.source_dataloader = create_dataloader(0.0, 0.0, "gaussian")
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
            self.target_dataloaders[_class] = create_dataloader(x, y, target_type)

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
            target_samples = self._target_samples[_class]
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

