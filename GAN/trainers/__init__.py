import os
import re
import tempfile
import traceback
from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Any, Type, List, Dict, Optional

import torch
from paramiko.ssh_exception import SSHException
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
    discriminator_optimizers: Dict[int, Optimizer] = {}

    use_same_batch_sizes: bool = False

    data_it: Iterator[Tensor]

    optimize_discriminator = True

    _artifact_log_failures = 0

    def __init__(self, config: Configuration):
        self.config = config

    @property
    def _mlflow_enabled(self):
        return self.config.train.mlflow.enabled

    def _log_mlflow_artifact_safe(self, local_path: str,
                                  artifact_path: Optional[str] = None) -> None:
        try:
            mlflow.log_artifact(local_path, artifact_path)
            self._artifact_log_failures = 0
        except SSHException:
            self._artifact_log_failures += 1
            traceback.print_exc()
            print(f"Current amount of artifact log failures: "
                  f"{self._artifact_log_failures}")

            if self._artifact_log_failures >= 5:
                raise Exception("Too many artifact log failures occurred.")

    @enable_mlflow_tracking_class('config')
    def train(self):
        device = self.config.runtime_options['device']
        self.dataset = datasets.load_dataset(
            self.config.dataset, device=device,
            num_workers=self.config.runtime_options['dataloader_num_workers'],
            batch_size=self.config.train.batch_size,
            latent_dimension=self.config.train.latent_dimension)

        self._initialize_networks()

        assert self.generator_network is not None

        self.generator_network.to(device)

        for network in self.discriminator_networks:
            network.to(device)

        models_path, images_path = self._prepare_directories()

        steps = 0
        epochs = 0
        if self.config.train.use_checkpoints:
            if self._mlflow_enabled:
                steps, epochs = self._load_mlflow_checkpoints()
            elif models_path:
                steps, epochs = self._load_checkpoints(models_path)

        # Need to initialize optimizers *after* loading models from checkpoint

        self.generator_optimizer = self._load_optimizer(
            self.config.optimizers.generator.type,
            self.generator_network.parameters(),
            **self.config.optimizers.generator.options)

        for i, discriminator in enumerate(self.discriminator_networks):
            if list(discriminator.parameters()):
                self.discriminator_optimizers[i] = \
                    self._load_optimizer(
                        self.config.optimizers.discriminator.type,
                        discriminator.parameters(),
                        **self.config.optimizers.discriminator.options)

        if not self.discriminator_optimizers:
            self.optimize_discriminator = False

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

        if steps == 0:
            img_paths = self.dataset.save_real_data(images_path, 'real_data')

            if self._mlflow_enabled:
                for path in img_paths:
                    self._log_mlflow_artifact_safe(path)

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

                        name = f'discriminator_{discriminator_index}_loss'
                        if self._mlflow_enabled:
                            mlflow.log_metric(name, discriminator_loss, steps)

                        self._check_tensor_nan_inf(loss, name)

                        self._optimize_discriminator(
                            loss,
                            self.discriminator_optimizers[discriminator_index])

            data_real: torch.Tensor = next(self.data_it)[0].to(device)
            batch_size = data_real.shape[0]

            step_batch_size_fake = batch_size if self.use_same_batch_sizes \
                else batch_size_fake

            loss = self._get_generator_loss(batch_size,
                                            step_batch_size_fake,
                                            data_real)

            generator_loss = loss.item()
            if self._mlflow_enabled:
                mlflow.log_metric('generator_loss', generator_loss, steps)

            print(f'Generator loss: {generator_loss}')

            self._check_tensor_nan_inf(loss, 'generator_loss')

            self._optimize_generator(loss, self.generator_optimizer)

            if steps % self.config.train.save_interval == 0 and steps > 0:
                print("Saving images and models")

                str_step_epoch = f'step_{steps:>06}_epoch_{epochs:>04}'

                with torch.no_grad():
                    img_paths = self.dataset.save_generated_data(
                        self.generator_network, images_path, str_step_epoch)

                    if self._mlflow_enabled:
                        for path in img_paths:
                            self._log_mlflow_artifact_safe(path, 'images')

                if models_path:
                    self._save_checkpoints(models_path, epochs, steps)

                if self._mlflow_enabled:
                    mlflow.pytorch.log_model(
                        self.generator_network,
                        f'models/generator_{str_step_epoch}/')

                    for i, disc in enumerate(self.discriminator_networks):
                        mlflow.pytorch.log_model(
                            disc,
                            f'models/discriminator_{i}_{str_step_epoch}/')

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

    def _prepare_directories(self) -> Tuple[str, str]:
        models_path: Optional[str] = None
        if self.config.train.output_directory:
            images_path = os.path.join(self.config.train.output_directory,
                                       'images')
            models_path = os.path.join(self.config.train.output_directory,
                                       'models')

            os.makedirs(self.config.train.output_directory, exist_ok=True)
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(models_path, exist_ok=True)
        else:
            self.tempdir = tempfile.TemporaryDirectory(prefix='gan-trainer-')
            images_path = self.tempdir.name

        return models_path, images_path

    def _load_checkpoints(self, checkpoints_path: str) -> Tuple[int, int]:
        print("Loading checkpoints")

        file_regex = re.compile(r'step_(\d+)_epoch_(\d+).pt')

        files = os.listdir(checkpoints_path)
        checkpoints = {}

        for file in files:
            match = file_regex.match(file)
            if not match:
                continue

            step = int(match.group(1))
            epoch = int(match.group(2))

            checkpoints[step] = (file, epoch)

        if not checkpoints:
            print("No checkpoints available to load.")
            return 0, 0

        load_step = max(checkpoints.keys())

        load_epoch = checkpoints[load_step][1]

        checkpoint_path = os.path.join(checkpoints_path,
                                       checkpoints[load_step][0])
        checkpoint_dict = torch.load(checkpoint_path)

        self.generator_network.load_state_dict(checkpoint_dict['generator'])
        self.generator_network.train()

        self.discriminator_networks[0].load_state_dict(
            checkpoint_dict['discriminator'])
        self.discriminator_networks[0].train()

        print(f"Loaded checkpoints at step {load_step} (epoch {load_epoch})")

        return load_step + 1, load_epoch

    def _save_checkpoints(self, checkpoints_path: str,
                          epoch: int, step: int) -> str:
        path = os.path.join(checkpoints_path, f'step_{step}_epoch_{epoch}.pt')
        torch.save({
            'generator': self.generator_network.state_dict(),
            'discriminator': self.discriminator_networks[0].state_dict(),
        }, path)

        return path

    def _load_mlflow_checkpoints(self) -> Tuple[int, int]:
        print("Loading checkpoints")

        artifact_generator_regex = re.compile(
            r'models/generator_step_(\d+)_epoch_(\d+)')
        checkpoints = {}

        run = mlflow.active_run()
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id, 'models')

        for artifact in artifacts:
            match = artifact_generator_regex.match(artifact.path)
            if not match:
                continue

            step = int(match.group(1))
            epoch = int(match.group(2))

            checkpoints[step] = (artifact, epoch)

        if not checkpoints:
            print("No checkpoints available to load.")
            return 0, 0

        load_step = max(checkpoints.keys())

        load_artifact, load_epoch = checkpoints[load_step]

        self.generator_network = mlflow.pytorch.load_model(
            f'{run.info.artifact_uri}/{load_artifact.path}')

        str_step_epoch = f'step_{load_step:>06}_epoch_{load_epoch:>04}'

        for i in range(len(self.discriminator_networks)):
            path = f'{run.info.artifact_uri}' \
                   f'/models/discriminator_{i}_{str_step_epoch}'
            self.discriminator_networks[i] = mlflow.pytorch.load_model(path)

        print(f"Loaded checkpoints at step {load_step} (epoch {load_epoch})")

        return load_step + 1, load_epoch

    def _check_tensor_nan_inf(self, tensor: Tensor, name: str):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise Exception(
                f"One or more values in {name} became NaN or +/-inf")

    @abstractmethod
    def _get_discriminator_loss(self,
                                discriminator_index: int,
                                batch_size_real: int,
                                batch_size_fake: int,
                                data_real: Tensor,
                                data_fake: Optional[Tensor] = None) -> Tensor:
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

    def _optimize_generator(self, loss: Tensor, optimizer: Optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
