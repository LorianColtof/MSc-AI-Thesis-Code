import os
from abc import ABC
from typing import Dict, List, Any

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from configuration import Dataset
from .base import AbstractBaseMulticlassDataset


class AbstractMnistBase(AbstractBaseMulticlassDataset, ABC):
    def __init__(self, classes: List[str], transform: Any,
                 channels: int, dataset_config: Dataset,
                 num_workers: int, batch_size: int):
        self.data_dimension = 28 * 28 * channels

        if not dataset_config.source_class:
            raise Exception("dataset.source_class is required "
                            "for multi-class datasets")

        self.source_class = dataset_config.source_class
        if self.source_class not in classes:
            raise Exception(f"Invalid source class: {self.source_class}. "
                            f"Allowed classes: {','.join(classes)}")

        self.target_dataloaders = {}
        for _class in classes:
            dataloader = DataLoader(
                datasets.ImageFolder(
                    os.path.join(dataset_config.directory, _class),
                    transform=transform),
                batch_size=batch_size, shuffle=True, pin_memory=True,
                drop_last=True, num_workers=num_workers)

            if _class == self.source_class:
                self.source_dataloader = dataloader
            else:
                self.target_dataloaders[_class] = dataloader

        # Make sure they are sorted nicely
        self._target_classes_sorted = [c for c in classes
                                       if c != dataset_config.source_class]

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
        return self._target_classes_sorted