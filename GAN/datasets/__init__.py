from configuration import Configuration
from datasets.base import AbstractBaseDataset
from datasets.celeba import CelebaDataset
from datasets.mnist import MnistDataset


_datasets = {
    'mnist': MnistDataset,
    'celeba': CelebaDataset
}


def load_dataset(config: Configuration) -> AbstractBaseDataset:
    try:
        dataset = _datasets[config.dataset.type]
    except KeyError:
        raise Exception(f"Dataset '{config.dataset.type}' does not exist")

    return dataset(config)
