import os
import re
from typing import Tuple

import torch
from torch import Tensor

import models
from configuration import Configuration
from trainers import AbstractBaseTrainer


class MaxSlicedWassersteinLossTrainer(AbstractBaseTrainer):
    use_same_batch_sizes = True

    def _initialize_networks(self):
        self.generator_network = models.load_model(
            self.config.models.generator.type,
            latent_dim=self.config.train.latent_dimension,
            output_dim=self.dataset.data_dimension,
            **self.config.models.generator.options)

        self.discriminator_network = models.load_model(
            self.config.models.discriminator.type,
            input_dim=self.dataset.data_dimension,
            **self.config.models.discriminator.options)

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

        self.discriminator_network.load_state_dict(
            checkpoint_dict['discriminator'])
        self.discriminator_network.train()

        print(f"Loaded checkpoints at step {load_step} (epoch {load_epoch})")

        return load_step + 1, load_epoch

    def _save_checkpoints(self, checkpoints_path: str, epoch: int,
                          step: int) -> None:
        torch.save({
            'generator': self.generator_network.state_dict(),
            'discriminator': self.discriminator_network.state_dict(),
        }, os.path.join(checkpoints_path, f'step_{step}_epoch_{epoch}.pt'))

    def _generate_data(self, batch_size: int, data_real: Tensor) -> Tensor:
        data_shape = data_real.shape[1:]
        z = torch.randn((batch_size, self.config.train.latent_dimension),
                        device=data_real.device)
        data_fake = self.generator_network(z).reshape(-1, *data_shape)

        return data_fake

    def _sample_directions(self, device):
        with torch.no_grad():
            data_sample = next(self.data_it)[0][:100].to(device)
            features_sample = self.discriminator_network(data_sample)
            features_norm = features_sample.norm(dim=1, p=2).unsqueeze(1)

            directions_sample = features_sample / features_norm

            return directions_sample

    def _get_discriminator_loss(self, batch_size_real: int,
                                batch_size_fake: int,
                                data_real: Tensor) -> Tensor:
        with torch.no_grad():
            data_fake = self._generate_data(batch_size_fake, data_real)

        # disc_real = self.discriminator_network(data_real)
        # disc_fake = self.discriminator_network(data_fake)
        # loss_discriminator = disc_real.mean() - disc_fake.mean()

        features_real = self.discriminator_network(data_real)
        features_fake = self.discriminator_network(data_fake)

        directions_sample = self._sample_directions(data_real.device)

        disc_real = directions_sample @ features_real.T
        disc_fake = directions_sample @ features_fake.T

        loss_discriminator = (disc_real.mean(dim=1) - disc_fake.mean(dim=1)).mean()

        return loss_discriminator

    def _get_generator_loss(self, batch_size_real: int, batch_size_fake: int,
                            data_real: Tensor) -> Tensor:
        data_fake = self._generate_data(batch_size_fake, data_real)

        features_real = self.discriminator_network(data_real)
        features_fake = self.discriminator_network(data_fake)

        directions_sample = self._sample_directions(data_real.device)

        disc_real = directions_sample @ features_real.T
        disc_fake = directions_sample @ features_fake.T

        disc_real_sorted = disc_real.sort(dim=1)[0]
        disc_fake_sorted = disc_fake.sort(dim=1)[0]

        loss = (disc_real_sorted - disc_fake_sorted).pow(
            2).sum(dim=1).mean()

        # disc_real = self.discriminator_network(data_real)
        # disc_fake = self.discriminator_network(data_fake)
        #
        # disc_real_sorted = disc_real.sort(dim=0)[0]
        # disc_fake_sorted = disc_fake.sort(dim=0)[0]
        #
        # loss = (disc_real_sorted - disc_fake_sorted).pow(2).sum()

        return loss
