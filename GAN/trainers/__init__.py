import os
from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Any, Type, List, Dict

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer
import mlflow
import mlflow.pytorch

import datasets
from configuration import Configuration
from utils.mlflow import enable_mlflow_tracking_class


class AbstractBaseTrainer(ABC):
    config: Configuration
    dataset: datasets.AbstractBaseDataset

    generator_network: Module
    discriminator_networks: List[Module] = []

    generator_optimizer: Optimizer
    discriminator_optimizers: Dict[Module, Optimizer] = {}

    use_same_batch_sizes: bool = False

    data_it: Iterator[Tensor]

    optimize_discriminator = True

    def __init__(self, config: Configuration):
        self.config = config

    @property
    def _mlflow_enabled(self):
        return self.config.train.mlflow.enabled

    @enable_mlflow_tracking_class('config')
    def train(self):
        device = self.config.runtime_options['device']
        self.dataset = datasets.load_dataset(
            self.config.dataset, device=device,
            batch_size=self.config.train.batch_size,
            latent_dimension=self.config.train.latent_dimension)

        self._initialize_networks()

        assert self.generator_network is not None

        self.generator_network.to(device)

        for network in self.discriminator_networks:
            network.to(device)

        self.generator_optimizer = self._load_optimizer(
            self.config.optimizers.generator.type,
            self.generator_network.parameters(),
            **self.config.optimizers.generator.options)

        for discriminator in self.discriminator_networks:
            if list(discriminator.parameters()):
                self.discriminator_optimizers[discriminator] = \
                    self._load_optimizer(
                        self.config.optimizers.discriminator.type,
                        discriminator.parameters(),
                        **self.config.optimizers.discriminator.options)

        if not self.discriminator_optimizers:
            self.optimize_discriminator = False

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

        while epochs <= self.config.train.maximum_epochs and \
                steps <= self.config.train.maximum_steps:
            print(f"Step {steps}")

            if self.optimize_discriminator:
                for discriminator_index, discriminator in \
                        enumerate(self.discriminator_networks):
                    for _ in range(self.config.train.critic_steps):
                        data_real: torch.Tensor = next(
                            self.data_it)[0].to(device)
                        batch_size = data_real.shape[0]

                        step_batch_size_fake = batch_size \
                            if self.use_same_batch_sizes else batch_size_fake

                        loss = self._get_discriminator_loss(
                            discriminator_index, batch_size,
                            step_batch_size_fake,
                            data_real)

                        discriminator_loss = loss.item()
                        print(f'Discriminator {discriminator_index} loss: '
                              f'{discriminator_loss}')

                        if self._mlflow_enabled:
                            mlflow.log_metric(
                                f'discriminator_{discriminator_index}_loss',
                                discriminator_loss)

                        self._optimize_discriminator(
                            loss, self.discriminator_optimizers[discriminator])

            data_real: torch.Tensor = next(self.data_it)[0].to(device)
            batch_size = data_real.shape[0]

            step_batch_size_fake = batch_size if self.use_same_batch_sizes \
                else batch_size_fake

            loss = self._get_generator_loss(batch_size,
                                            step_batch_size_fake,
                                            data_real)

            generator_loss = loss.item()
            if self._mlflow_enabled:
                mlflow.log_metric('generator_loss', generator_loss)

            print(f'Generator loss: {generator_loss}')

            self._optimize_generator(loss)

            if steps % self.config.train.save_interval == 0 and steps > 0:
                print("Saving images and models")

                with torch.no_grad():
                    img_path = self.dataset.save_generated_data(
                        self.generator_network, images_path, steps, epochs)

                    if self._mlflow_enabled:
                        mlflow.log_artifact(img_path, 'images')

                self._save_checkpoints(models_path, epochs, steps)

                if self._mlflow_enabled:
                    mlflow.pytorch.log_model(
                        self.generator_network,
                        f'models/generator_step_{steps}_epoch_{epochs}/')

                    for i, disc in enumerate(self.discriminator_networks):
                        mlflow.pytorch.log_model(
                            disc, f'models/discriminator_{i}'
                                  f'_step_{steps}_epoch_{epochs}/')

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
                          epoch: int, step: int) -> str:
        pass

    @abstractmethod
    def _get_discriminator_loss(self,
                                discriminator_index: int,
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

    def _optimize_discriminator(self, loss: Tensor, optimizer: Optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _optimize_generator(self, loss: Tensor):
        self.generator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()
