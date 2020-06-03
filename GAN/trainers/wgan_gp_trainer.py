import os
import re
from typing import Tuple

import torch
from torch import Tensor

import models
from trainers import AbstractBaseTrainer
from utils import gradient_penalty


class WassersteinGPLossTrainer(AbstractBaseTrainer):
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

    def _save_checkpoints(self, checkpoints_path: str, epoch: int,
                          step: int) -> str:
        path = os.path.join(checkpoints_path, f'step_{step}_epoch_{epoch}.pt')

        torch.save({
            'generator': self.generator_network.state_dict(),
            'discriminator': self.discriminator_networks[0].state_dict(),
        }, path)

        return path

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

        penalty = gradient_penalty(self.discriminator_networks[0],
                                   data_real, data_fake)
        disc_generated = self.discriminator_networks[0](data_fake)
        disc_real = -self.discriminator_networks[0](data_real)
        loss_discriminator = (-(disc_generated.mean() + disc_real.mean()) +
                              penalty)

        return loss_discriminator

    def _get_generator_loss(self, batch_size_real: int, batch_size_fake: int,
                            data_real: Tensor) -> Tensor:
        data_fake = self._generate_data(batch_size_fake, data_real)

        disc_generated = self.discriminator_networks[0](data_fake)
        disc_real = -self.discriminator_networks[0](data_real)
        loss_generator = disc_generated.mean() + disc_real.mean()

        return loss_generator
