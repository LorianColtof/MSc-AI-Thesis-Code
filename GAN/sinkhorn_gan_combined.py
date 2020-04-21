import os
import re
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Tuple, Type
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import Parameter
from torch.optim import Optimizer
from torchvision import datasets, transforms
from torchvision.utils import save_image
import PIL.Image as Image
from ot.smooth import smooth_ot_dual

import models
from models import GoodGenerator, GoodDiscriminator, \
    MnistGenerator, MnistDiscriminator
import configuration


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# -------------For WGAN------------
def dist(x, y=None, q=2, p=1):
    Nx = x.shape[0]
    if y is None:
        y = x
    Ny = y.shape[0]

    # Lq norm
    D = torch.norm(x.contiguous().view(Nx, -1)[:, None]
                   - y.contiguous().view(Ny, -1), dim=2, p=q)
    return (D) ** (p) / p


def objective_balanced(label_real: torch.Tensor, label_fake: torch.Tensor,
                       C: torch.Tensor, epsilon: float) -> torch.Tensor:
    val0 = torch.mean(label_fake)
    val1 = torch.mean(label_real)

    tmp0 = (-C + (label_fake + torch.t(label_real)))
    val_reg = epsilon * torch.mean(torch.exp(tmp0 / epsilon))

    val = val0 + val1 - val_reg

    return val


def objective_unbalanced(label_real: torch.Tensor, label_fake: torch.Tensor,
                         C: torch.Tensor, epsilon: float,
                         tau: float) -> torch.Tensor:
    val_fake = tau * torch.mean(1 - torch.exp(-label_fake / tau))
    val_real = tau * torch.mean(1 - torch.exp(-label_real / tau))

    tmp0 = (-C + (label_fake + label_real.T))
    val_reg = epsilon * torch.mean(torch.exp(tmp0 / epsilon))

    val = val_fake + val_real - val_reg
    return val


def c_e_transform(y: torch.Tensor, C: torch.Tensor,
                  epsilon: float = 1) -> torch.Tensor:
    return softmin((C - y), epsilon=epsilon)[:None]


def c_e_tau_transform(y: torch.Tensor, C: torch.Tensor,
                      epsilon: float = 1, tau: float = 1):
    return aprox(softmin(C - y, epsilon), epsilon, tau)


def aprox(p: torch.Tensor, epsilon: float, tau: float) -> torch.Tensor:
    return (tau / (tau + epsilon)) * p


def softmin(X: torch.tensor, epsilon: float = 1) -> torch.Tensor:
    Y = -X / epsilon
    Ymax = torch.max(Y, 0)[0][:, None]
    return -epsilon * (
            Ymax + torch.log(torch.mean(torch.exp(Y - Ymax.t()), 0))[:,
                   None])


def load_optimizers(config: configuration.Configuration,
                    generator_params: Iterator[Parameter],
                    discriminator_params: Iterator[Parameter]) \
        -> Tuple[Optimizer, Optimizer]:

    try:
        generator_optimizer: Type[Optimizer] = getattr(
            torch.optim, config.optimizers.generator.type)
    except AttributeError:
        raise Exception(
            f"Optimizer type '{config.optimizers.generator.type}' "
            "does not exist.")

    try:
        discriminator_optimizer: Type[Optimizer] = getattr(
            torch.optim, config.optimizers.discriminator.type)
    except AttributeError:
        raise Exception(
            f"Optimizer type '{config.optimizers.discriminator.type}' "
            "does not exist.")

    return (generator_optimizer(params=generator_params,
                                **config.optimizers.generator.options),
            discriminator_optimizer(params=discriminator_params,
                                    **config.optimizers.discriminator.options))


class ModelTrainingConfiguration(ABC):
    dataloader: torch.utils.data.DataLoader

    latent_dimension: int
    data_dimension: int

    q: float
    p: float

    subtract_ot_bias: bool = False

    @abstractmethod
    def dual_variable_transform(
            self, label_real: torch.Tensor,
            label_fake: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def objective_function(self, label_real: torch.Tensor,
                           label_fake: torch.Tensor,
                           cost_matrix: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def save_generated_data(self, generator_network: torch.nn.Module,
                            data_fake: torch.Tensor, images_path: str,
                            steps: int, epochs: int) -> None:
        pass


class BalancedEntropyConfigurationBase(ModelTrainingConfiguration, ABC):
    epsilon: float
    gamma: float

    def __init__(self, config: configuration.Configuration):
        self.epsilon = config.loss.options['epsilon']
        self.gamma = 1 / self.epsilon

    def dual_variable_transform(self, label_real: torch.Tensor,
                                cost_matrix: torch.Tensor) -> torch.Tensor:
        return c_e_transform(label_real, cost_matrix, epsilon=self.epsilon)

    def objective_function(self, label_real: torch.Tensor,
                           label_fake: torch.Tensor,
                           cost_matrix: torch.Tensor) -> torch.Tensor:
        return objective_balanced(label_real, label_fake, cost_matrix,
                                  epsilon=self.epsilon)


class UnbalancedEntropyConfigurationBase(ModelTrainingConfiguration, ABC):
    epsilon: float
    tau: float

    def __init__(self, config: configuration.Configuration):
        self.epsilon = config.loss.options['epsilon']
        self.tau = config.loss.options['tau']

    def dual_variable_transform(self, label_real: torch.Tensor,
                                cost_matrix: torch.Tensor) -> torch.Tensor:
        return c_e_tau_transform(label_real, cost_matrix,
                                 epsilon=self.epsilon, tau=self.tau)

    def objective_function(self, label_real: torch.Tensor,
                           label_fake: torch.Tensor,
                           cost_matrix: torch.Tensor) -> torch.Tensor:
        return objective_unbalanced(label_real, label_fake, cost_matrix,
                                    epsilon=self.epsilon, tau=self.tau)


class MnistConfigurationBase(ModelTrainingConfiguration, ABC):
    p = 2
    q = 2

    def __init__(self, config: configuration.Configuration):
        self.latent_dimension = 100

        # self.subtract_ot_bias = True

        batch_size = 128
        lr_generator = 0.0002
        lr_discriminator = 1e-4

        # load data
        self.dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(config.dataset.directory, 'mnist'),
                           train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])),
            batch_size=batch_size, shuffle=True, pin_memory=True)

        self.data_dimension = 28 * 28

    def save_generated_data(self, generator_network: torch.nn.Module,
                            data_fake: torch.Tensor, images_path: str,
                            steps: int, epochs: int) -> None:
        save_image(data_fake[:25].reshape(-1, 1, 28, 28),
                   os.path.join(images_path,
                                'epoch_{}_step_{}.png'.format(epochs, steps)),
                   nrow=5, normalize=True)


class MnistBalancedConfiguration(MnistConfigurationBase,
                                 BalancedEntropyConfigurationBase):
    def __init__(self, config: configuration.Configuration):
        MnistConfigurationBase.__init__(self, config)
        BalancedEntropyConfigurationBase.__init__(self, config)


class MnistUnbalancedConfiguration(MnistConfigurationBase,
                                   UnbalancedEntropyConfigurationBase):
    def __init__(self, config: configuration.Configuration):
        MnistConfigurationBase.__init__(self, config)
        UnbalancedEntropyConfigurationBase.__init__(self, config)


class CelebaConfigurationBase(ModelTrainingConfiguration, ABC):
    p = 2
    q = 2

    _source_samples_plot: torch.Tensor

    def __init__(self, config: configuration.Configuration):
        DSOURCE = 128
        NBATCH = 1

        self.latent_dimension = DSOURCE

        crop_size = 108
        re_size = 64
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size,
                         offset_width:offset_width + crop_size]

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(crop),
             transforms.ToPILImage(),
             transforms.Resize(size=(re_size, re_size),
                               interpolation=Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

        self.data_dimension = re_size ** 2

        # wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets
        # /celeba.zip
        # Make sure to change to correct path!
        imagenet_data = datasets.ImageFolder(config.dataset.directory,
                                             transform=transform)
        self.dataloader = torch.utils.data.DataLoader(imagenet_data,
                                                      batch_size=NBATCH,
                                                      shuffle=True,
                                                      num_workers=0)

        # Fix latent samples for visualization purposes
        self._source_samples_plot = torch.randn(
            (5 * 5, DSOURCE), device=config.runtime_options['device'])

    def save_generated_data(self, generator_network: torch.nn.Module,
                            data_fake: torch.Tensor, images_path: str,
                            steps: int, epochs: int) -> None:
        NC = 3
        IMGSIZE = 64

        fig = plt.figure(figsize=(5, 5))
        samples_plot = generator_network(
            self._source_samples_plot).cpu().detach()

        for k in range(5 * 5):
            plt.subplot(5, 5, k + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            imshow(samples_plot[k].reshape(NC, IMGSIZE, IMGSIZE))
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0,
                            top=1)

        plt.savefig(os.path.join(
            images_path, 'epoch_{}_step_{}.png'.format(epochs, steps)),
            dpi=75)
        plt.close(fig)


class CelebaBalancedConfiguration(CelebaConfigurationBase,
                                  BalancedEntropyConfigurationBase):
    def __init__(self, config: configuration.Configuration):
        CelebaConfigurationBase.__init__(self, config)
        BalancedEntropyConfigurationBase.__init__(self, config)


class CelebaUnbalancedConfiguration(CelebaConfigurationBase,
                                    UnbalancedEntropyConfigurationBase):
    def __init__(self, config: configuration.Configuration):
        CelebaConfigurationBase.__init__(self, config)
        UnbalancedEntropyConfigurationBase.__init__(self, config)


def load_checkpoints(generator_network: torch.nn.Module,
                     discriminator_network: torch.nn.Module,
                     models_path: str) -> Tuple[int, int]:
    print("Loading checkpoints")

    file_regex = re.compile(
        r'(discriminator|generator)_step_(\d+)_epoch_(\d+).pt')

    files = os.listdir(models_path)
    discriminator_checkpoints = {}
    generator_checkpoints = {}

    for file in files:
        match = file_regex.match(file)
        if not match:
            continue

        _type = match.group(1)
        step = int(match.group(2))
        epoch = int(match.group(3))

        if _type == 'discriminator':
            discriminator_checkpoints[step] = (file, epoch)
        else:
            generator_checkpoints[step] = (file, epoch)

    if not discriminator_checkpoints or not generator_checkpoints:
        print("No checkpoints available to load.")
        return 0, 0

    max_step_discriminator = max(discriminator_checkpoints.keys())
    max_step_generator = max(discriminator_checkpoints.keys())

    load_step: int

    if max_step_discriminator != max_step_generator:
        print("WARNING: found generator and discriminator checkpoints "
              "at different steps")
        load_step = min(max_step_discriminator, max_step_generator)
    else:
        load_step = max_step_discriminator

    load_epoch = discriminator_checkpoints[load_step][1]

    generator_checkpoint_path = os.path.join(
        models_path, generator_checkpoints[load_step][0])
    generator_network.load_state_dict(
        torch.load(generator_checkpoint_path))
    generator_network.train()

    discriminator_checkpoint_path = os.path.join(
        models_path, discriminator_checkpoints[load_step][0])
    discriminator_network.load_state_dict(
        torch.load(discriminator_checkpoint_path))
    discriminator_network.train()

    print(f"Loaded checkpoints at step {load_step} (epoch {load_epoch})")

    return load_step + 1, load_epoch


def train_regularized_ot_GAN(
        config: configuration.Configuration,
        train_config: ModelTrainingConfiguration,
        max_epochs: int,
        max_steps: Optional[int] = None,
        save_interval: int = 200,
        num_train_discriminator: int = 1) -> None:

    p = train_config.p
    q = train_config.q

    device = config.runtime_options['device']

    generator_network, discriminator_network = models.load_models(
        config,
        dict(latent_dim=train_config.latent_dimension,
             output_dim=train_config.data_dimension),
        dict(input_dim=train_config.data_dimension))

    generator_network.to(device)
    discriminator_network.to(device)

    generator_optimizer, discriminator_optimizer = load_optimizers(
        config, generator_network.parameters(),
        discriminator_network.parameters())

    images_path = os.path.join(config.train.output_directory, 'images')
    models_path = os.path.join(config.train.output_directory, 'models')

    os.makedirs(config.train.output_directory, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    steps, epochs = load_checkpoints(generator_network, discriminator_network,
                                     models_path)

    cost_matrix_cross: torch.Tensor
    cost_matrix_real: torch.Tensor
    cost_matrix_fake: torch.Tensor

    def data_iterator() -> Iterator[torch.Tensor]:
        nonlocal epochs

        while True:
            print(f"Epoch {epochs}")
            for data in train_config.dataloader:
                yield data

            epochs += 1

    def lq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.norm(x.unsqueeze(1) - y, dim=2, p=q) ** p) / q

    def compute_cost_matrices(data_real: torch.Tensor,
                              data_fake: torch.Tensor):
        nonlocal cost_matrix_cross
        nonlocal cost_matrix_real
        nonlocal cost_matrix_fake

        cost_matrix_cross = lq_dist(data_fake, data_real)

        if train_config.subtract_ot_bias:
            cost_matrix_real = lq_dist(data_real, data_real)
            cost_matrix_fake = lq_dist(data_fake, data_fake)

    def get_loss() -> torch.Tensor:
        label_fake = discriminator_network(data_fake)
        label_real = discriminator_network(data_real)

        label_real_transformed_fake = train_config.dual_variable_transform(
            label_fake, cost_matrix_cross)

        loss_val = train_config.objective_function(
            label_real_transformed_fake, label_fake, cost_matrix_cross)
        if train_config.subtract_ot_bias:
            label_real_transformed = train_config.dual_variable_transform(
                label_real, cost_matrix_real)
            label_fake_transformed = train_config.dual_variable_transform(
                label_fake, cost_matrix_fake)
            bias_real = train_config.objective_function(
                label_real, label_real_transformed, cost_matrix_real)
            bias_fake = train_config.objective_function(
                label_fake, label_fake_transformed, cost_matrix_fake)

            # print(bias_real.item())
            # print(bias_fake.item())

            loss_val -= 0.5 * (bias_real + bias_fake)

        return loss_val

    data_it = data_iterator()
    while epochs < max_epochs:
        print(f"Step {steps}")

        data_real: torch.Tensor = next(data_it)[0].to(device)
        batch_size = data_real.size(0)
        data_real = data_real.reshape(batch_size, -1)

        with torch.no_grad():
            z = torch.randn((batch_size, train_config.latent_dimension),
                            device=device)
            data_fake = generator_network(z).reshape(
                batch_size, -1)

            compute_cost_matrices(data_real, data_fake)

        for _ in range(num_train_discriminator):
            loss = -get_loss()

            print(f'Discriminator loss: {loss.item()}')

            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()

        data_fake = generator_network(z).reshape(batch_size, -1)

        compute_cost_matrices(data_real, data_fake)

        loss = get_loss()

        print(f'Generator loss: {loss.item()}')

        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()

        if steps % save_interval == 0 and steps > 0:
            print("Saving images and models")
            train_config.save_generated_data(generator_network, data_fake,
                                             images_path, steps, epochs)

            torch.save(generator_network.state_dict(),
                       os.path.join(
                           models_path,
                           f'generator_step_{steps}_epoch_{epochs}.pt'))
            torch.save(discriminator_network.state_dict(),
                       os.path.join(
                           models_path,
                           f'discriminator_step_{steps}_epoch_{epochs}.pt'))

        steps += 1

        if max_steps and steps > max_steps:
            print("Reached maximum amount of steps. Quitting.")
            break


def train_mnist(config: configuration.Configuration):
    if config.loss.type == 'unbalanced':
        train_configuration = MnistUnbalancedConfiguration(config)
    elif config.loss.type == 'balanced':
        train_configuration = MnistBalancedConfiguration(config)
    else:
        raise Exception(f'Unknown loss type: {config.loss.type}')

    train_regularized_ot_GAN(config, train_configuration,
                             max_epochs=config.train.maximum_epochs,
                             max_steps=config.train.maximum_steps,
                             num_train_discriminator=config.train.critic_steps,
                             save_interval=config.train.save_interval)


def train_celeba(config: configuration.Configuration):
    if config.loss.type == 'unbalanced':
        train_configuration = CelebaUnbalancedConfiguration(config)
    elif config.loss.type == 'balanced':
        train_configuration = CelebaBalancedConfiguration(config)
    else:
        raise Exception(f'Unknown loss type: {config.loss.type}')

    train_regularized_ot_GAN(config, train_configuration,
                             max_epochs=config.train.maximum_epochs,
                             max_steps=config.train.maximum_steps,
                             num_train_discriminator=config.train.critic_steps,
                             save_interval=config.train.save_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=argparse.FileType('r'),
                        required=True)

    args = parser.parse_args()

    config = configuration.load_configuration(args.config_file)

    if torch.cuda.is_available():
        print("Using CUDA")
        config.runtime_options['device'] = torch.device('cuda')
    else:
        print("CUDA is not available, falling back to CPU")
        config.runtime_options['device'] = torch.device('cpu')

    if config.dataset.type == 'mnist':
        train_mnist(config)
    elif config.dataset.type == 'celeba':
        train_celeba(config)
    else:
        raise Exception(f'Unknown dataset type: {config.dataset.type}')


if __name__ == "__main__":
    main()
