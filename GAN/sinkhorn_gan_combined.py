import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, Iterator
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import PIL.Image as Image

from nn import GoodGenerator, GoodDiscriminator, \
    MnistGenerator, MnistDiscriminator


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


class ModelTrainingConfiguration(ABC):
    generator_network: torch.nn.Module
    discriminator_network: torch.nn.Module
    generator_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer

    dataloader: torch.utils.data.DataLoader
    output_directory: str

    latent_dimension: int
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
    def save_generated_data(self, data_fake: torch.Tensor, images_path: str,
                            steps: int, epochs: int) -> None:
        pass


class BalancedEntropyConfigurationBase(ModelTrainingConfiguration, ABC):
    epsilon: float

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

    def __init__(self, dataset_dir: str, output_directory: str):
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device('cuda')
        else:
            print("CUDA is not available, falling back to CPU")
            device = torch.device('cpu')

        self.latent_dimension = 100
        self.output_directory = output_directory

        # self.subtract_ot_bias = True

        batch_size = 128
        lr_generator = 0.0002
        lr_discriminator = 1e-4

        # load data
        self.dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(dataset_dir, 'mnist'),
                           train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])),
            batch_size=batch_size, shuffle=True, pin_memory=True)

        img_dim = 28 * 28

        # Initialize models and optimizers
        self.generator_network = MnistGenerator(
            self.latent_dimension, img_dim).to(device)
        self.discriminator_network = MnistDiscriminator(img_dim).to(device)

        self.generator_optimizer = torch.optim.Adam(
            self.generator_network.parameters(), lr=lr_generator)
        self.discriminator_optimizer = torch.optim.RMSprop(
            self.discriminator_network.parameters(), lr=lr_discriminator)

    def save_generated_data(self, data_fake: torch.Tensor, images_path: str,
                            steps: int, epochs: int) -> None:
        save_image(data_fake[:25].reshape(-1, 1, 28, 28),
                   os.path.join(images_path,
                                'epoch_{}_step_{}.png'.format(epochs, steps)),
                   nrow=5, normalize=True)


class MnistBalancedConfiguration(MnistConfigurationBase,
                                 BalancedEntropyConfigurationBase):
    def __init__(self, dataset_dir: str, output_directory: str,
                 epsilon: float):
        MnistConfigurationBase.__init__(self, dataset_dir, output_directory)

        self.epsilon = epsilon


class MnistUnbalancedConfiguration(MnistConfigurationBase,
                                   UnbalancedEntropyConfigurationBase):
    def __init__(self, dataset_dir: str, output_directory: str,
                 epsilon: float, tau: float):
        MnistConfigurationBase.__init__(self, dataset_dir, output_directory)

        self.epsilon = epsilon
        self.tau = tau


class CelebaConfigurationBase(ModelTrainingConfiguration, ABC):
    p = 2
    q = 2

    _source_samples_plot: torch.Tensor

    def __init__(self, dataset_directory: str, output_directory: str):
        dtype = torch.cuda.FloatTensor
        device = torch.cuda.current_device()

        DSOURCE = 128
        NBATCH = 64

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
             transforms.Scale(size=(re_size, re_size),
                              interpolation=Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

        # wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets
        # /celeba.zip
        # Make sure to change to correct path!
        imagenet_data = datasets.ImageFolder(dataset_directory,
                                             transform=transform)
        self.dataloader = torch.utils.data.DataLoader(imagenet_data,
                                                      batch_size=NBATCH,
                                                      shuffle=True,
                                                      num_workers=0)

        self.generator_network = GoodGenerator().to(device)
        self.discriminator_network = GoodDiscriminator().to(device)

        self.generator_optimizer = torch.optim.Adam(
            self.generator_network.parameters(), lr=1e-5, betas=(0.5, .999))
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator_network.parameters(), lr=1e-4,
            betas=(0.5, .999))

        self.output_directory = output_directory

        # Fix latent samples for visualization purposes
        self._source_samples_plot = torch.randn((5 * 5, DSOURCE)).type(dtype)

    def save_generated_data(self, data_fake: torch.Tensor, images_path: str,
                            steps: int, epochs: int) -> None:
        NC = 3
        IMGSIZE = 64

        fig = plt.figure(figsize=(5, 5))
        samples_plot = self.generator_network(
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
    def __init__(self, dataset_dir: str, output_directory: str,
                 epsilon: float):
        CelebaConfigurationBase.__init__(self, dataset_dir, output_directory)

        self.epsilon = epsilon


class CelebaUnbalancedConfiguration(CelebaConfigurationBase,
                                    UnbalancedEntropyConfigurationBase):
    def __init__(self, dataset_dir: str, output_directory: str,
                 epsilon: float, tau: float):
        CelebaConfigurationBase.__init__(self, dataset_dir, output_directory)

        self.epsilon = epsilon
        self.tau = tau


def train_regularized_ot_GAN(
        configuration: ModelTrainingConfiguration,
        max_epochs: int,
        max_steps: Optional[int] = None,
        save_interval: int = 200,
        num_train_discriminator: int = 1) -> None:
    p = configuration.p
    q = configuration.q

    device = next(configuration.generator_network.parameters()).device

    images_path = os.path.join(configuration.output_directory, 'images')
    models_path = os.path.join(configuration.output_directory, 'models')

    os.makedirs(configuration.output_directory, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    steps = 0
    epochs = 0

    cost_matrix_cross: torch.Tensor
    cost_matrix_real: torch.Tensor
    cost_matrix_fake: torch.Tensor

    def data_iterator() -> Iterator[torch.Tensor]:
        nonlocal epochs

        while True:
            print(f"Epoch {epochs}")
            for data in configuration.dataloader:
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

        if configuration.subtract_ot_bias:
            cost_matrix_real = lq_dist(data_real, data_real)
            cost_matrix_fake = lq_dist(data_fake, data_fake)

    def get_loss() -> torch.Tensor:
        label_fake = configuration.discriminator_network(data_fake)
        label_real = configuration.discriminator_network(data_real)

        label_real_transformed_fake = configuration.dual_variable_transform(
            label_fake, cost_matrix_cross)

        loss_val = configuration.objective_function(
            label_real_transformed_fake, label_fake, cost_matrix_cross)
        if configuration.subtract_ot_bias:
            label_real_transformed = configuration.dual_variable_transform(
                label_real, cost_matrix_real)
            label_fake_transformed = configuration.dual_variable_transform(
                label_fake, cost_matrix_fake)
            bias_real = configuration.objective_function(
                label_real, label_real_transformed, cost_matrix_real)
            bias_fake = configuration.objective_function(
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
            z = torch.randn((batch_size, configuration.latent_dimension),
                            device=device)
            data_fake = configuration.generator_network(z).reshape(
                batch_size, -1)

            compute_cost_matrices(data_real, data_fake)

        for _ in range(num_train_discriminator):
            loss = -get_loss()

            print(f'Discriminator loss: {loss.item()}')

            configuration.discriminator_optimizer.zero_grad()
            loss.backward()
            configuration.discriminator_optimizer.step()

        data_fake = configuration.generator_network(z).reshape(batch_size, -1)

        compute_cost_matrices(data_real, data_fake)

        loss = get_loss()

        print(f'Generator loss: {loss.item()}')

        configuration.generator_optimizer.zero_grad()
        loss.backward()
        configuration.generator_optimizer.step()

        if steps % save_interval == 0 and steps > 0:
            print("Saving images and models")
            configuration.save_generated_data(data_fake, images_path,
                                              steps, epochs)

            torch.save(configuration.generator_network.state_dict(),
                       os.path.join(
                           models_path,
                           f'generator_step_{steps}_epoch_{epochs}.pt'))
            torch.save(configuration.discriminator_network.state_dict(),
                       os.path.join(
                           models_path,
                           f'discriminator_step_{steps}_epoch_{epochs}.pt'))

        steps += 1

        if max_steps and steps > max_steps:
            print("Reached maximum amount of steps. Quitting.")
            break


def train_mnist(args):
    eps = args.epsilon
    tau = args.tau

    if args.ot_type == 'unbalanced':
        configuration = MnistUnbalancedConfiguration(
            args.dataset_dir, args.output_dir, eps, tau)
    else:
        configuration = MnistBalancedConfiguration(
            args.dataset_dir, args.output_dir, eps)

    N_epochs = 10000
    N_critic = 1
    N_steps = 3000

    train_regularized_ot_GAN(
        configuration, max_epochs=N_epochs, max_steps=N_steps,
        num_train_discriminator=N_critic, save_interval=args.save_interval)


def train_celeba(args):
    eps = args.epsilon
    tau = args.tau

    if args.ot_type == 'unbalanced':
        configuration = CelebaUnbalancedConfiguration(
            args.dataset_dir, args.output_dir, eps, tau)
    else:
        configuration = CelebaBalancedConfiguration(
            args.dataset_dir, args.output_dir, eps)

    N_epochs = 10000
    N_critic = 1
    N_steps = 3000

    train_regularized_ot_GAN(configuration,
                             max_epochs=N_epochs, max_steps=N_steps,
                             num_train_discriminator=N_critic,
                             save_interval=args.save_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="Directory which contains the dataset.")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to save images and models to.")
    parser.add_argument('--save-interval', type=int, default=200, metavar='S',
                        help="Save images and models every S steps.")
    parser.add_argument('--dataset', required=True, choices=[
        'mnist', 'celeba'], help='Selects which dataset type to use.')
    parser.add_argument('--ot-type', required=True, choices=[
        'balanced', 'unbalanced'], help='Selects which OT type to use.')
    parser.add_argument('--epsilon', default=1, type=float)
    parser.add_argument('--tau', default=100, type=float)

    args = parser.parse_args()

    print(f"eps={args.epsilon} tau={args.tau}")

    if args.dataset == 'mnist':
        train_mnist(args)
    else:
        train_celeba(args)


if __name__ == "__main__":
    main()
