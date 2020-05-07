import os
from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Any, Type

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer

import datasets
from configuration import Configuration


class AbstractBaseTrainer(ABC):
    config: Configuration
    dataset: datasets.AbstractBaseDataset

    generator_network: Module
    discriminator_network: Module

    generator_optimizer: Optimizer
    discriminator_optimizer: Optimizer

    use_same_batch_sizes: bool = False

    data_it: Iterator[Tensor]

    def __init__(self, config: Configuration):
        self.config = config

    def train(self):
        device = self.config.runtime_options['device']
        self.dataset = datasets.load_dataset(self.config)

        self._initialize_networks()

        assert self.generator_network is not None
        assert self.discriminator_network is not None

        self.generator_network.to(device)
        self.discriminator_network.to(device)

        self.generator_optimizer = self._load_optimizer(
            self.config.optimizers.generator.type,
            self.generator_network.parameters(),
            **self.config.optimizers.generator.options)

        self.discriminator_optimizer = self._load_optimizer(
            self.config.optimizers.discriminator.type,
            self.discriminator_network.parameters(),
            **self.config.optimizers.discriminator.options)

        images_path = os.path.join(self.config.train.output_directory,
                                   'images')
        models_path = os.path.join(self.config.train.output_directory,
                                   'models')

        os.makedirs(self.config.train.output_directory, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)

        if self.config.train.use_checkpoints:
            steps, epochs = self._load_checkpoints(models_path)
        else:
            steps = 0
            epochs = 0

        def data_iterator() -> Iterator[torch.Tensor]:
            nonlocal epochs

            while True:
                print(f"Epoch {epochs}")
                for data in self.dataset.dataloader:
                    yield data

                epochs += 1

        self.data_it = data_iterator()

        batch_size_fake = self.config.train.batch_size_fake \
            if self.config.train.batch_size_fake and \
            not self.use_same_batch_sizes \
            else self.config.train.batch_size

        while epochs < self.config.train.maximum_epochs and \
                steps < self.config.train.maximum_steps:
            print(f"Step {steps}")

            for _ in range(self.config.train.critic_steps):
                data_real: torch.Tensor = next(self.data_it)[0].to(device)
                batch_size = data_real.shape[0]

                step_batch_size_fake = batch_size if self.use_same_batch_sizes \
                    else batch_size_fake

                loss = self._get_discriminator_loss(batch_size,
                                                    step_batch_size_fake,
                                                    data_real)

                print(f'Discriminator loss: {loss.item()}')

                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()

                self.discriminator_network.normalize_final_linear()

            data_real: torch.Tensor = next(self.data_it)[0].to(device)
            batch_size = data_real.shape[0]

            step_batch_size_fake = batch_size if self.use_same_batch_sizes \
                else batch_size_fake

            loss = self._get_generator_loss(batch_size,
                                            step_batch_size_fake,
                                            data_real)

            print(f'Generator loss: {loss.item()}')

            self.generator_optimizer.zero_grad()
            loss.backward()
            self.generator_optimizer.step()

            if steps % self.config.train.save_interval == 0 and steps > 0:
                print("Saving images and models")

                with torch.no_grad():
                    self.dataset.save_generated_data(self.generator_network,
                                                     images_path, steps, epochs)

                self._save_checkpoints(models_path, epochs, steps)

            steps += 1

    @abstractmethod
    def _initialize_networks(self):
        pass

    @staticmethod
    def _load_optimizer(optim_type: str,
                        model_params: Iterator[Parameter],
                        **kwargs: Any) -> Optimizer:
        try:
            optimizer: Type[Optimizer] = getattr(
                torch.optim, optim_type)
        except AttributeError:
            raise Exception(
                f"Optimizer type '{optim_type}' "
                "does not exist.")

        return optimizer(params=model_params, **kwargs)

    @abstractmethod
    def _load_checkpoints(self, checkpoints_path: str) -> Tuple[int, int]:
        pass

    @abstractmethod
    def _save_checkpoints(self, checkpoints_path: str,
                          epoch: int, step: int) -> None:
        pass

    @abstractmethod
    def _get_discriminator_loss(self,
                                batch_size_real: int,
                                batch_size_fake: int,
                                data_real: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _get_generator_loss(self,
                            batch_size_real: int,
                            batch_size_fake: int,
                            data_real: Tensor) -> Tensor:
        pass
