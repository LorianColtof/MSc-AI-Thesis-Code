from typing import Iterator

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

import datasets
import models
from configuration import Configuration
from trainers import AbstractBaseTrainer


class MaxSlicedWassersteinLossTrainer(AbstractBaseTrainer):
    use_same_batch_sizes = True

    def __init__(self, config: Configuration):
        super().__init__(config)

        if config.loss.type != 'swd':
            raise Exception("Loss type can only be 'swd' when using "
                            "mswd_trainer or sampled_swd_trainer")

    def _initialize_networks(self):
        self.generator_network = models.load_model(
            self.config.models.generator.type,
            latent_dim=self.config.train.latent_dimension,
            output_dim=self.dataset.data_dimension,
            **self.config.models.generator.options)

        self.discriminator_networks.append(models.load_model(
            self.config.models.discriminator.type,
            input_dim=self.dataset.data_dimension,
            **self.config.models.discriminator.options))

    def _generate_data(self, batch_size: int, data_real: Tensor) -> Tensor:
        data_shape = data_real.shape[1:]
        z = torch.randn((batch_size, self.config.train.latent_dimension),
                        device=data_real.device)
        data_fake = self.generator_network(z).reshape(-1, *data_shape)

        return data_fake

    def _get_discriminator_loss(self,
                                discriminator_index: int,
                                batch_size_real: int,
                                batch_size_fake: int,
                                data_real: Tensor) -> Tensor:
        with torch.no_grad():
            data_fake = self._generate_data(batch_size_fake, data_real)

        disc_real = self.discriminator_networks[0](data_real)
        disc_fake = self.discriminator_networks[0](data_fake)
        loss_discriminator = disc_real.mean() - disc_fake.mean()

        return loss_discriminator

    def _get_generator_loss(self, batch_size_real: int, batch_size_fake: int,
                            data_real: Tensor) -> Tensor:
        data_fake = self._generate_data(batch_size_fake, data_real)

        disc_real = self.discriminator_networks[0](data_real)
        disc_fake = self.discriminator_networks[0](data_fake)

        disc_real_sorted = disc_real.sort(dim=0)[0]
        disc_fake_sorted = disc_fake.sort(dim=0)[0]

        loss = (disc_real_sorted - disc_fake_sorted).pow(2).sum()

        return loss

    def _optimize_discriminator(self, optim: Optimizer, loss: Tensor):
        super()._optimize_discriminator(optim, loss)

        self.discriminator_networks[0].normalize_final_linear()


class SampledSlicedWassersteinLossTrainer(MaxSlicedWassersteinLossTrainer):
    def __init__(self, config: Configuration):
        super().__init__(config)

        dataset_directions = datasets.load_dataset(
            config.dataset, device=config.runtime_options['device'],
            num_workers=config.runtime_options['dataloader_num_workers'],
            batch_size=config.loss.options['direction_sample_batch_size'],
            latent_dimension=config.train.latent_dimension,
            drop_last=True)

        def data_iterator() -> Iterator[torch.Tensor]:
            while True:
                for data in dataset_directions.dataloader:
                    yield data

        self.data_iterator_directions = data_iterator()

        self.num_projections = self.config.loss.options[
            'direction_sample_batch_size']

        if not self.config.loss.options['sample_source'] in {'random', 'data'}:
            raise Exception("sample_source should be one of: random, data")
        self.sample_random = (self.config.loss.options['sample_source']
                              == 'random')

    def _normalize_directions(self, directions: torch.Tensor) -> torch.Tensor:
        features_norm = directions.norm(dim=1, p=2).unsqueeze(1)

        # Remove feature vectors with a very small norm to
        # (hopefully) prevent numerical issues.
        features_mask = features_norm > 1e-11
        features_sample = directions[features_mask.squeeze(), :]
        features_norm = features_norm[features_mask].unsqueeze(1)

        directions_sample = features_sample / features_norm

        return directions_sample

    def _sample_directions_data(self, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            data_sample = next(self.data_iterator_directions)[0].to(device)
            
            features_sample = self.discriminator_networks[0](
                data_sample).reshape(self.num_projections, -1)

            return self._normalize_directions(features_sample)

    def _sample_directions_random(self, dim: int, device: torch.device) \
            -> torch.Tensor:
        with torch.no_grad():
            features_sample = torch.randn((self.num_projections, dim),
                                          device=device)

            return self._normalize_directions(features_sample)

    def _get_discriminator_loss(self,
                                discriminator_index: int,
                                batch_size_real: int,
                                batch_size_fake: int,
                                data_real: Tensor) -> Tensor:
        with torch.no_grad():
            data_fake = self._generate_data(batch_size_fake, data_real)

        features_real = self.discriminator_networks[0](data_real).reshape(
            batch_size_real, -1)
        features_fake = self.discriminator_networks[0](data_fake).reshape(
            batch_size_fake, -1)

        if self.sample_random:
            directions_sample = self._sample_directions_random(
                features_real.shape[1], data_real.device)
        else:
            directions_sample = self._sample_directions_data(data_real.device)

        disc_real = directions_sample @ features_real.T
        disc_fake = directions_sample @ features_fake.T

        loss_discriminator = (disc_real.mean(dim=1)
                              - disc_fake.mean(dim=1)).mean()

        return loss_discriminator

    def _get_generator_loss(self, batch_size_real: int, batch_size_fake: int,
                            data_real: Tensor) -> Tensor:
        data_fake = self._generate_data(batch_size_fake, data_real)

        features_real = self.discriminator_networks[0](data_real).reshape(
            batch_size_real, -1)
        features_fake = self.discriminator_networks[0](data_fake).reshape(
            batch_size_fake, -1)

        if self.sample_random:
            directions_sample = self._sample_directions_random(
                features_real.shape[1], data_real.device)
        else:
            directions_sample = self._sample_directions_data(data_real.device)

        disc_real = directions_sample @ features_real.T
        disc_fake = directions_sample @ features_fake.T

        disc_real_sorted = disc_real.sort(dim=1, descending=True)[0]
        disc_fake_sorted = disc_fake.sort(dim=1, descending=True)[0]

        loss = (disc_real_sorted
                - disc_fake_sorted).pow(2).mean()

        return loss


