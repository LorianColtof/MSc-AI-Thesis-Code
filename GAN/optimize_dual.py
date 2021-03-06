import argparse
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import torch
from torch.distributions.normal import Normal
from torch.distributions.binomial import Binomial

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
            trainer.use_dual_discriminator_networks:
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

    return -loss.detach()


@torch.no_grad()
def compute_potentials_ot_gan(trainer: OTLossTrainer,
                              data_1: torch.Tensor,
                              data_2: torch.Tensor,
                              sample_data_2: Callable[[int], torch.Tensor],
                              num_transform_approx_samples: int = 100000) \
        -> Tuple[torch.Tensor, torch.Tensor]:

    potential_2_values = trainer.discriminator_networks[0](data_2)
    if trainer.use_dual_discriminator_networks:
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


@torch.no_grad()
def compute_potentials(trainer: AbstractBaseTrainer,
                       data_1: torch.Tensor,
                       data_2: torch.Tensor,
                       sample_data_2: Callable[[int], torch.Tensor],
                       num_transform_approx_samples: int = 100000) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(trainer, OTLossTrainer):
        return compute_potentials_ot_gan(
            trainer, data_1, data_2, sample_data_2,
            num_transform_approx_samples)
    elif isinstance(trainer, WassersteinGPLossTrainer):
        potential_1_values = trainer.discriminator_networks[0](data_1)
        potential_2_values = -trainer.discriminator_networks[0](data_2)

        return potential_1_values, potential_2_values


def get_cost_matrix(trainer: AbstractBaseTrainer,
                    data_1: torch.Tensor,
                    data_2: torch.Tensor) -> torch.Tensor:
    if isinstance(trainer, OTLossTrainer):
        return trainer._data_distance(data_1, data_2)
    elif isinstance(trainer, WassersteinGPLossTrainer):
        return (data_1 - data_2).norm(p=1)


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

    device = config.runtime_options['device']
    data_dim = 1

    # Example using two 1D Gaussians

    # mu_1 = torch.tensor(0.25, device=device)
    # mu_2 = torch.tensor(0.8,  device=device)
    # sigma = torch.tensor(0.06, device=device)

    # dist_1 = Normal(mu_1, sigma)
    # dist_2 = Normal(mu_2, sigma)

    # Example using two 1D Binomial distributions

    n = 100
    p_1 = torch.tensor(0.25, device=device)
    p_2 = torch.tensor(0.8, device=device)

    dist_1 = Binomial(n, probs=p_1)
    dist_2 = Binomial(n, probs=p_2)

    def sample_1(size: int) -> torch.Tensor:
        return dist_1.sample(torch.Size([size])).reshape(size, data_dim)

    def sample_2(size: int) -> torch.Tensor:
        return dist_2.sample(torch.Size([size])).reshape(size, data_dim)

    run_optimization(trainer, data_dim, sample_1, sample_2)

    plt.plot(range(config.train.maximum_steps + 1), losses)
    plt.show()

    # These are the values for which we want
    # to compute the potential values
    values_dist = torch.arange(0, n, device=device,
                               dtype=torch.float32).reshape(-1, data_dim)

    # Get potential values
    potentials = compute_potentials(trainer, values_dist, values_dist,
                                    sample_2)

    # Compute cost matrix
    cost_matrix = get_cost_matrix(trainer, values_dist, values_dist)

    # Compute coupling matrix
    coupling_matrix = torch.exp(
        (-cost_matrix + potentials[0] + potentials[1].T)
        / trainer.config.loss.options['epsilon'])

    plt.imshow(torch.log(coupling_matrix + 1e-5))
    plt.show()

if __name__ == "__main__":
    main()
