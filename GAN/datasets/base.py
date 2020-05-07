from abc import ABC, abstractmethod

import torch


class AbstractBaseDataset(ABC):
    dataloader: torch.utils.data.DataLoader
    data_dimension: int

    @abstractmethod
    def save_generated_data(self, generator_network: torch.nn.Module,
                            images_path: str, steps: int, epochs: int) -> None:
        pass


