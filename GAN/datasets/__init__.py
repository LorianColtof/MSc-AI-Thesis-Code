from typing import Any

from configuration import Configuration, Dataset
from datasets.base import AbstractBaseDataset
from datasets.celeba import CelebaDataset
from datasets.mnist import MnistDataset


_datasets = {
    'mnist': MnistDataset,
    'celeba': CelebaDataset
}


def load_dataset(dataset_config: Dataset, **kwargs: Any) -> AbstractBaseDataset:
    try:
        dataset = _datasets[dataset_config.type]
    except KeyError:
        raise Exception(f"Dataset '{dataset_config.type}' does not exist")

    return dataset(dataset_config=dataset_config, **kwargs)
