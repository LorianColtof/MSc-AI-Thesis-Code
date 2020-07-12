from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from torch.utils.data import DataLoader


class AbstractBaseMulticlassDataset(ABC):
    source_class: str
    source_dataloader: DataLoader
    target_dataloaders: Dict[str, DataLoader]
    data_dimension: int

    # @abstractmethod
    # def save_generated_data(self, generator_network: torch.nn.Module,
    #                         images_path: str, filename: str) -> str:
    #     pass

    def get_dataloader(self, _class: str) -> DataLoader:
        if _class == self.source_class:
            return self.source_dataloader
        return self.target_dataloaders[_class]

    @property
    def target_classes(self) -> List[str]:
        return sorted(list(self.target_dataloaders.keys()))

    @property
    def classes(self) -> List[str]:
        return [self.source_class] + self.target_classes

