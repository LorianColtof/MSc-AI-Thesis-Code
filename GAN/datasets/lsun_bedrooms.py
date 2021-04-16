import os
from typing import List

import torch
import torch.utils.data as utilsdata
from torchvision.datasets import LSUN
from torchvision.transforms import transforms
from torchvision.utils import save_image

from configuration import Dataset
from datasets import AbstractBaseDataset


class LsunBedroomsDataset(AbstractBaseDataset):
    _source_samples_plot: torch.Tensor

    _sample_image_size = 10

    def __init__(self, dataset_type: str, dataset_config: Dataset, num_workers: int,
                 device: torch.device,
                 batch_size: int, latent_dimension: int,
                 drop_last: bool = False):
        self.data_dimension = 3 * 256 * 256

        self.dataloader = utilsdata.DataLoader(
            LSUN(dataset_config.directory,
                 classes=["bedroom_" + dataset_type],
                 transform=transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(256),
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
        save_image(data_fake.reshape(-1, 3, 256, 256),
                   img_path, nrow=self._sample_image_size, normalize=True)

        return [img_path]

    def save_real_data(self, images_path: str, filename: str) -> List[str]:
        data_real = next(iter(
            self.dataloader))[0][:self._sample_image_size ** 2]
        img_path = os.path.join(images_path, f'{filename}.png')
        save_image(data_real, img_path,
                   nrow=self._sample_image_size, normalize=True)

        return [img_path]


class LsunBedroomsValidationDataset(LsunBedroomsDataset):
    def __init__(self, dataset_config: Dataset, num_workers: int,
                 device: torch.device, batch_size: int, latent_dimension: int):
        super().__init__("val", dataset_config, num_workers, device, batch_size,
                         latent_dimension)


class LsunBedroomsTrainingDataset(LsunBedroomsDataset):
    def __init__(self, dataset_config: Dataset, num_workers: int,
                 device: torch.device, batch_size: int, latent_dimension: int):
        super().__init__("train", dataset_config, num_workers, device, batch_size,
                         latent_dimension)
