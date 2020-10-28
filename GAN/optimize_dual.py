import argparse
from typing import Callable, Tuple, List

import matplotlib.pyplot as plt
import torch
from torch.distributions.normal import Normal

import configuration
import models
from trainers import AbstractBaseTrainer
from trainers.ot_trainer import OTLossTrainer
from trainers.wgan_gp_trainer import WassersteinGPLossTrainer

losses = []


def run_optimization(trainer: AbstractBaseTrainer,
                     sample_data_dim: int,
                     sample_data_real: Callable[[int], torch.Tensor],
                     sample_data_fake: Callable[[int], torch.Tensor]) \
                        -> torch.Tensor:
    device = trainer.config.runtime_options['device']

    # We cannot just use _initialize_networks since we don't have a dataset
    # to get the shapes from, so initialize discriminator_networks directly

    num_discriminators = 1
    if isinstance(trainer, OTLossTrainer) and \
            trainer.config.train.use_dual_critic_networks:
        num_discriminators += 1

    for _ in range(num_discriminators):
        network = models.load_model(
            trainer.config.models.discriminator.type,
            input_dim=sample_data_dim,
            **trainer.config.models.discriminator.options)

        network.to(device)
        trainer.discriminator_networks.append(network)

    # Skip learning cost function for now
    trainer.cost_function = lambda x: x

    for i, discriminator in enumerate(trainer.discriminator_networks):
        if list(discriminator.parameters()):
            trainer.discriminator_optimizers[i] = \
                trainer._load_optimizer(
                    trainer.config.optimizers.discriminator.type,
                    discriminator.parameters(),
                    **trainer.config.optimizers.discriminator.options)

    if not trainer.discriminator_optimizers:
        trainer.optimize_discriminator = False

    batch_size = trainer.config.train.batch_size
    batch_size_fake = trainer.config.train.batch_size_fake \
        if trainer.config.train.batch_size_fake and \
        not trainer.use_same_batch_sizes \
        else batch_size

    steps = 0
    loss: torch.Tensor

    while steps <= trainer.config.train.maximum_steps:
        print(f"Step {steps}")

        if trainer.optimize_discriminator:
            for discriminator_index, discriminator in \
                    enumerate(trainer.discriminator_networks):
                for _ in range(trainer.config.train.critic_steps):

                    data_real = sample_data_real(
                        trainer.config.train.batch_size)
                    data_fake = sample_data_fake(batch_size_fake)

                    loss = trainer._get_discriminator_loss(
                        discriminator_index, batch_size,
                        batch_size, data_real, data_fake)

                    discriminator_loss = loss.item()
                    print(f'Discriminator {discriminator_index} loss: '
                          f'{discriminator_loss}')

                    # You can do something with the loss here if you want
                    losses.append(-loss.item())

                    name = f'discriminator_{discriminator_index}_loss'
                    trainer._check_tensor_nan_inf(loss, name)

                    trainer._optimize_discriminator(
                        loss,
                        trainer.discriminator_optimizers[discriminator_index])

        steps += 1

    return -loss


def compute_potentials_ot_gan(trainer: OTLossTrainer,
                              sample_data_dim: int,
                              data_1: torch.Tensor,
                              data_2: torch.Tensor,
                              sample_data_2: Callable[[int], torch.Tensor],
                              num_transform_approx_samples: int = 100000) \
        -> Tuple[torch.Tensor, torch.Tensor]:

    potential_2_values = trainer.discriminator_networks[0](data_2)
    if trainer.config.train.use_dual_critic_networks:
        potential_1_values = trainer.discriminator_networks[1](data_1)
    elif trainer.use_same_discriminator_as_potentials:
        potential_1_values = trainer.discriminator_networks[0](data_1)
    else:
        # For distribution 1 we need to compute the (c,e)-transform
        # We need to have a good amount of samples from distribution 2 to
        # approximate the (c,e)-transform
        samples_2 = sample_data_2(num_transform_approx_samples)

        potential_2_sample_values = trainer.discriminator_networks[0](
            samples_2)

        cost_matrix = trainer._data_distance(samples_2, data_1)
        potential_1_values = trainer.ot_loss_helper.dual_variable_transform(
            potential_2_sample_values, cost_matrix)

    return potential_1_values, potential_2_values


def compute_potentials(trainer: AbstractBaseTrainer,
                       sample_data_dim: int,
                       data_1: torch.Tensor,
                       data_2: torch.Tensor,
                       sample_data_2: Callable[[int], torch.Tensor],
                       num_transform_approx_samples: int = 100000) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(trainer, OTLossTrainer):
        return compute_potentials_ot_gan(
            trainer, sample_data_dim, data_1,
            data_2, sample_data_2, num_transform_approx_samples)
    elif isinstance(trainer, WassersteinGPLossTrainer):
        potential_1_values = trainer.discriminator_networks[0](data_1)
        potential_2_values = -trainer.discriminator_networks[0](data_2)

        return potential_1_values, potential_2_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=argparse.FileType('r'),
                        required=True)
    parser.add_argument('--no-cuda', action='store_true')

    args = parser.parse_args()

    config = configuration.load_configuration(args.config_file)

    # Or you could create a Configuration object directly here,
    # albeit a bit cumbersome

    config.runtime_options['device'] = torch.device('cpu')

    if torch.cuda.is_available():
        if args.no_cuda:
            print("CUDA is disabled. Using CPU.")
        else:
            print("Using CUDA")
            config.runtime_options['device'] = torch.device('cuda')
    else:
        print("CUDA is not available, falling back to CPU")

    config.runtime_options['config_filename'] = args.config_file.name

    torch.random.manual_seed(42)

    trainer: AbstractBaseTrainer

    if config.train.type == 'ot_gan':
        trainer = OTLossTrainer(config)
    elif config.train.type == 'wgan_gp':
        trainer = WassersteinGPLossTrainer(config)
    else:
        raise Exception(f'Unsupported training type: {config.train.type}')

    # Example using two 1D Gaussians
    mu_1 = torch.tensor(0.25, device=config.runtime_options['device'])
    mu_2 = torch.tensor(0.8,  device=config.runtime_options['device'])
    sigma = torch.tensor(0.06, device=config.runtime_options['device'])

    normal_1 = Normal(mu_1, sigma)
    normal_2 = Normal(mu_2, sigma)

    data_dim = 1

    def sample_1(size: int) -> torch.Tensor:
        return normal_1.sample(torch.Size([size])).reshape(size, data_dim)

    def sample_2(size: int) -> torch.Tensor:
        return normal_2.sample(torch.Size([size])).reshape(size, data_dim)

    loss = run_optimization(trainer, data_dim, sample_1, sample_2)

    # These are the values for which we want
    # to compute the potential values
    values_dist_1 = sample_1(10)
    values_dist_2 = sample_2(10)

    potentials = compute_potentials(trainer, data_dim,
                                    values_dist_1, values_dist_2, sample_2)

    plt.plot(range(config.train.maximum_steps + 1), losses)
    plt.show()


if __name__ == "__main__":
    main()
