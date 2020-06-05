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
