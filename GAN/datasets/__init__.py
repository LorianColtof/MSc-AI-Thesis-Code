from typing import Any, Union

from configuration import Dataset
from datasets.base import AbstractBaseDataset
from datasets.celeba import CelebaDataset
from datasets.mnist import MnistDataset
from datasets.gaussian_mixture import GaussianMixtureDataset
from datasets.multiclass.base import AbstractBaseMulticlassDataset
from datasets.multiclass.colored_mnist import ColoredMnistDataset
from datasets.multiclass.rotated_mnist import RotatedMnistDataset
from datasets.multiclass.celeba import CelebaDataset as MulticlassCelebaDataset

_datasets = {
    'mnist': MnistDataset,
    'celeba': CelebaDataset,
    'gaussian_mixture': GaussianMixtureDataset,
    'multiclass.rotated_mnist': RotatedMnistDataset,
    'multiclass.colored_mnist': ColoredMnistDataset,
    'multiclass.celeba': MulticlassCelebaDataset
}


def load_dataset(dataset_config: Dataset, **kwargs: Any) -> \
        Union[AbstractBaseDataset, AbstractBaseMulticlassDataset]:
    try:
        dataset = _datasets[dataset_config.type]
    except KeyError:
        raise Exception(f"Dataset '{dataset_config.type}' does not exist")

    return dataset(dataset_config=dataset_config, **kwargs)
