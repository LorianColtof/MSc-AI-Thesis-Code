import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Iterator
import itertools

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import mlflow
import mlflow.pytorch

import datasets
from configuration import Configuration
from trainers import AbstractBaseTrainer
from datasets.multiclass.base import AbstractBaseMulticlassDataset
from utils.mlflow import enable_mlflow_tracking_class


class AbstractMultimarginalBaseTrainer(AbstractBaseTrainer, ABC):
    dataset: AbstractBaseMulticlassDataset

    generator_networks: List[Module]
    generator_optimizers: Dict[int, Optimizer] = {}
    generator_class_to_index: Dict[str, int] = {}

    use_one_generator_optimizer: bool = False

    encoder_network: Module
    optimize_encoder = True

    current_step: int

    def __init__(self, config: Configuration):
        super().__init__(config)

    def _get_discriminator_loss(self,
                                discriminator_index: int,
                                batch_size_real: int,
                                batch_size_fake: int,
                                data_real: Tensor) -> Tensor:
        assert False, "Should never be called"

    def _get_generator_loss(self,
                            batch_size_real: int,
                            batch_size_fake: int,
                            data_real: Tensor) -> Tensor:
        assert False, "Should never be called"

    @enable_mlflow_tracking_class('config')
    def train(self):
        device = self.config.runtime_options['device']
        self.dataset = datasets.load_dataset(
            self.config.dataset, device=device,
            num_workers=self.config.runtime_options['dataloader_num_workers'],
            batch_size=self.config.train.batch_size,
            latent_dimension=self.config.train.latent_dimension)

        if not isinstance(self.dataset, AbstractBaseMulticlassDataset):
            raise Exception("Dataset needs to be a multi-class dataset")

        self._initialize_networks()

        assert self.generator_networks and \
            self.generator_networks[0] is not None

        models_path, images_path = self._prepare_directories()

        step = 0

        if self.config.train.use_checkpoints:
            if self._mlflow_enabled:
                step = self._load_mlflow_checkpoints()
            elif models_path:
                step = self._load_checkpoints(models_path)

        # Need to initialize optimizers *after* loading models from checkpoint

        for i, network in enumerate(self.generator_networks):
            network.to(device)

            if not self.use_one_generator_optimizer:
                optimizer = super()._load_optimizer(
                    self.config.optimizers.generator.type,
                    network.parameters(),
                    **self.config.optimizers.generator.options)
                self.generator_optimizers[i] = optimizer

        if self.use_one_generator_optimizer:
            optimizer = super()._load_optimizer(
                self.config.optimizers.generator.type,
                itertools.chain(self.encoder_network.parameters(),
                                *(network.parameters()
                                for network in self.generator_networks)),
                **self.config.optimizers.generator.options)

            self.generator_optimizers[0] = optimizer

        for i, network in enumerate(self.discriminator_networks):
            network.to(device)

            if list(network.parameters()):
                optimizer = super()._load_optimizer(
                    self.config.optimizers.discriminator.type,
                    network.parameters(),
                    **self.config.optimizers.discriminator.options)
                self.discriminator_optimizers[i] = optimizer

        if not self.discriminator_optimizers:
            self.optimize_discriminator = False

        if self.encoder_network:
            self.encoder_network.to(device)
            self.optimize_encoder = True


        self.current_step = step

        def data_iterator(_class: str) -> Iterator[Tensor]:
            dataloader = self.dataset.get_dataloader(_class)
            while True:
                for data in dataloader:
                    yield data

        source_data_iterator = data_iterator(self.dataset.source_class)
        target_data_iterators = {
            _class: data_iterator(_class)
            for _class in self.dataset.target_classes
        }

        self.generator_class_to_index = {
            _class: i for i, _class in enumerate(self.dataset.target_classes)
        }

        sample_images: Tensor

        with torch.no_grad():
            sample_images_list = []
            num_samples = 0
            while num_samples < self.config.train.num_samples:
                sample_images_batch = next(source_data_iterator)[0]
                sample_images_list.append(sample_images_batch)
                num_samples += sample_images_batch.size(0)

            sample_images = torch.cat(
                sample_images_list, dim=0)[:self.config.train.num_samples]

        while self.current_step <= self.config.train.maximum_steps:
            print(f"Step {self.current_step}")

            if self.optimize_discriminator:
                for discriminator_index, discriminator in \
                        enumerate(self.discriminator_networks):
                    for _ in range(self.config.train.critic_steps):
                        source_data: Tensor = next(
                            source_data_iterator)[0].to(device)
                        target_data: List[Tensor] = [
                            next(target_data_iterators[_class])[0].to(device)
                            for _class in self.dataset.target_classes
                        ]
                        batch_size = source_data.shape[0]

                        loss = self._get_multiclass_discriminator_loss(
                            discriminator_index, batch_size,
                            source_data, target_data)

                        self._log_loss(
                            f'Discriminator {discriminator_index} loss',
                            f'discriminator_{discriminator_index}_loss',
                            loss)

                        self._optimize_discriminator(
                            loss,
                            self.discriminator_optimizers[discriminator_index])

            data_real: Tensor = next(source_data_iterator)[0].to(device)
            batch_size = data_real.shape[0]

            self._train_multiclass_generators(batch_size, data_real,
                                              target_data_iterators)

            if self.current_step % self.config.train.save_interval == 0 \
                    and self.current_step > 0:
                print("Saving images and models")

                str_step_epoch = f'step_{self.current_step:>06}'

                for generator in self.generator_networks:
                    generator.eval()

                self.encoder_network.eval()

                with torch.no_grad():
                    embeddings = self.encoder_network(
                        sample_images.to(device))
                    generated_imgs = {
                        _class: self.generator_networks[
                                     self.generator_class_to_index[
                                         _class]](embeddings)
                        for _class in self.dataset.target_classes
                    }

                    img_paths = self.dataset.save_generated_data(
                        sample_images, generated_imgs,
                        images_path, str_step_epoch)

                    if self._mlflow_enabled:
                        for path in img_paths:
                            self._log_mlflow_artifact_safe(path, 'images')

                for generator in self.generator_networks:
                    generator.train()

                self.encoder_network.train()

                if models_path:
                    self._save_checkpoints(models_path, 0, self.current_step)

                if self._mlflow_enabled:
                    if self.optimize_encoder:
                        self._log_mlflow_model_safe(
                            self.encoder_network,
                            f'models/source_encoder_{str_step_epoch}/')

                    for i, generator in enumerate(self.generator_networks):
                        self._log_mlflow_model_safe(
                            generator,
                            f'models/generator_{i}_{str_step_epoch}/')

                    for i, disc in enumerate(self.discriminator_networks):
                        self._log_mlflow_model_safe(
                            disc,
                            f'models/discriminator_{i}_{str_step_epoch}/')

            self.current_step += 1

    def _load_checkpoints(self, checkpoints_path: str) -> int:
        # TODO: implement
        print("Checkpoint loading not implemented yet")
        return 0

    def _load_mlflow_checkpoints(self) -> int:
        print("Loading checkpoints")

        artifact_generator_regex = re.compile(
            r'models/generator_0_step_(\d+)')
        checkpoints = {}

        run = mlflow.active_run()
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id, 'models')

        for artifact in artifacts:
            match = artifact_generator_regex.match(artifact.path)
            if not match:
                continue

            step = int(match.group(1))

            checkpoints[step] = artifact

        if not checkpoints:
            print("No checkpoints available to load.")
            return 0

        load_step = max(checkpoints.keys())
        load_artifact = checkpoints[load_step]

        self.generator_networks[0] = mlflow.pytorch.load_model(
            f'{run.info.artifact_uri}/{load_artifact.path}')

        str_step_epoch = f'step_{load_step:>06}'

        for i in range(1, len(self.generator_networks)):
            path = f'{run.info.artifact_uri}' \
                   f'/models/generator_{i}_{str_step_epoch}'
            self.generator_networks[i] = mlflow.pytorch.load_model(path)

        for i in range(len(self.discriminator_networks)):
            path = f'{run.info.artifact_uri}' \
                   f'/models/discriminator_{i}_{str_step_epoch}'
            self.discriminator_networks[i] = mlflow.pytorch.load_model(path)

        if self.optimize_encoder:
            path = f'{run.info.artifact_uri}' \
                   f'/models/source_encoder_{str_step_epoch}'
            self.encoder_network = mlflow.pytorch.load_model(path)

        print(f"Loaded checkpoints at step {load_step}")
        return load_step + 1

    def _save_checkpoints(self, checkpoints_path: str,
                          epoch: int, step: int) -> str:
        path = os.path.join(checkpoints_path, f'step_{step}.pt')

        save_dict = {
            **{
                f'generator{i}': self.generator_networks[i].state_dict()
                for i in range(len(self.generator_networks))
            },
            **{
                f'discriminator{i}':
                    self.discriminator_networks[i].state_dict()
                for i in range(len(self.discriminator_networks))
            }
        }

        if self.optimize_encoder:
            save_dict['source_encoder'] = self.encoder_network.state_dict()

        torch.save(save_dict, path)

        return path

    @abstractmethod
    def _get_multiclass_discriminator_loss(
            self, discriminator_index: int,
            batch_size: int, data_source: Tensor,
            data_targets: List[Tensor]) -> Tensor:
        pass

    @abstractmethod
    def _train_multiclass_generators(
            self, batch_size: int, data_source: Tensor,
            target_data_iterators: Dict[str, Iterator[Tensor]]):
        pass

    def _log_loss(self, printable_name: str,
                  metric_name: str, loss_tensor: Tensor):
        loss = loss_tensor.item()

        print(f'{printable_name}: {loss}')

        if self._mlflow_enabled:
            mlflow.log_metric(metric_name, loss, self.current_step)

        self._check_tensor_nan_inf(loss_tensor, metric_name)
