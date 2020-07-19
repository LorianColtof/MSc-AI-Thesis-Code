from typing import Any, Union

from configuration import Dataset
from datasets.base import AbstractBaseDataset
from datasets.celeba import CelebaDataset
from datasets.mnist import MnistDataset
from datasets.multiclass.base import AbstractBaseMulticlassDataset
from datasets.multiclass.colored_mnist import ColoredMnistDataset
from datasets.multiclass.rotated_mnist import RotatedMnistDataset

_datasets = {
    'mnist': MnistDataset,
    'celeba': CelebaDataset,
    'multiclass.rotated_mnist': RotatedMnistDataset,
    'multiclass.colored_mnist': ColoredMnistDataset
}


def load_dataset(dataset_config: Dataset, **kwargs: Any) -> \
        Union[AbstractBaseDataset, AbstractBaseMulticlassDataset]:
    try:
        dataset = _datasets[dataset_config.type]
    except KeyError:
        raise Exception(f"Dataset '{dataset_config.type}' does not exist")

    return dataset(dataset_config=dataset_config, **kwargs)
