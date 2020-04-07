import os
from typing import Callable, Iterable, Optional, Iterator
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


def train_regularized_ot_GAN(
        generator_network: torch.nn.Module,
        discriminator_network: torch.nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        latent_dimension: int,
        dual_variable_transform: Callable[[torch.Tensor, torch.Tensor],
                                          torch.Tensor],
        objective_function: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        save_generated_data_callback: Callable[[torch.Tensor, str, int, int],
                                               None],
        dataloader: torch.utils.data.DataLoader,
        output_directory: str,
        max_epochs: int,
        max_steps: Optional[int] = None,
        save_interval: int = 200,
        num_train_discriminator: int = 1) -> None:

    p = 2
    q = 2

    device = next(generator_network.parameters()).device

    images_path = os.path.join(output_directory, 'images')
    models_path = os.path.join(output_directory, 'models')

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    steps = 0
    epochs = 0

    def data_iterator() -> Iterator[torch.Tensor]:
        nonlocal epochs

        while True:
            print(f"Epoch {epochs}")
            for data in dataloader:
                yield data

            epochs += 1

    def lq_dist(x, y):
        return (torch.norm(x.unsqueeze(1) - y, dim=2, p=q) ** p) / q

    data_it = data_iterator()
    while epochs < max_epochs:
        print(f"Step {steps}")

        data_real: torch.Tensor = next(data_it)[0].to(device)
        batch_size = data_real.size(0)
        data_real = data_real.reshape(batch_size, -1)

        with torch.no_grad():
            z = torch.randn((batch_size, latent_dimension), device=device)
            data_fake = generator_network(z).reshape(batch_size, -1)
            C = lq_dist(data_fake, data_real)

        for _ in range(num_train_discriminator):
            label_fake = discriminator_network(data_fake)
            label_real = dual_variable_transform(label_fake, C)

            loss = -objective_function(label_real, label_fake, C)
            print(f'Discriminator loss: {loss.item()}')

            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()

        data_fake = generator_network(z).reshape(batch_size, -1)
        C = lq_dist(data_fake, data_real)

        label_fake = discriminator_network(data_fake)
        label_real = dual_variable_transform(label_fake, C)

        loss = objective_function(label_real, label_fake, C)
        print(f'Generator loss: {loss.item()}')

        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()

        if steps % save_interval == 0 and steps > 0:
            print("Saving images and models")
            save_generated_data_callback(data_fake, images_path, steps, epochs)

            torch.save(generator_network.state_dict(), os.path.join(
                           models_path,
                           f'generator_step_{steps}_epoch_{epochs}.pt'))
            torch.save(discriminator_network.state_dict(), os.path.join(
                models_path,
                f'discriminator_step_{steps}_epoch_{epochs}.pt'))

        steps += 1

        if max_steps and steps >= max_steps:
            print("Reached maximum amount of steps. Quitting.")
            break


def train_mnist(args):
    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device('cuda')
    else:
        print("CUDA is not available, falling back to CPU")
        device = torch.device('cpu')

    batch_size = 128
    latent_dim = 100
    lr_generator = 0.0002
    lr_discriminator = 1e-4

    N_epochs = 10000
    N_critic = 1
    epsilon = 1
    tau = 100

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(args.dataset_dir, 'mnist'),
                       train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    img_dim = 28 * 28

    # Initialize models and optimizers
    generator = MnistGenerator(latent_dim, img_dim)
    discriminator = MnistDiscriminator(img_dim)

    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=lr_generator)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(),
                                      lr=lr_discriminator)

    def _c_epsilon_transform(y, C):
        return c_e_transform(y, C, epsilon=epsilon)

    def _c_epsilon_tau_transform(y, C):
        return c_e_tau_transform(y, C, epsilon=epsilon, tau=tau)

    def _objective(label_real, label_fake, C):
        val0 = torch.mean(label_fake)
        val1 = torch.mean(label_real)

        tmp0 = (-C + (label_fake + torch.t(label_real)))
        val_reg = epsilon * torch.mean(torch.exp(tmp0 / epsilon))

        val = val0 + val1 - val_reg
        return val

    def _objective_unbalanced(label_real, label_fake, C):
        val_fake = tau * torch.mean(1 - torch.exp(-label_fake / tau))
        val_real = tau * torch.mean(1 - torch.exp(-label_real / tau))

        tmp0 = (-C + (label_fake + label_real.T))
        val_reg = epsilon * torch.mean(torch.exp(tmp0 / epsilon))

        val = val_fake + val_real - val_reg
        return val

    def save_images(data_fake: torch.Tensor,
                    images_path: str, steps: int, epochs: int):
        save_image(data_fake[:25].reshape(-1, 1, 28, 28),
                   os.path.join(images_path,
                                'epoch_{}_step_{}.png'.format(epochs, steps)),
                   nrow=5, normalize=True)

    train_regularized_ot_GAN(
        generator, discriminator, optimizer_G, optimizer_D,
        latent_dim,
        _c_epsilon_tau_transform, _objective_unbalanced,
        # _c_epsilon_transform, _objective,
        save_images, dataloader, args.output_dir,
        max_epochs=N_epochs, max_steps=3000,
        num_train_discriminator=N_critic,
        save_interval=args.save_interval)


def train_celeba(args):
    torch.backends.cudnn.enabled = False
    dtype = torch.cuda.FloatTensor
    device = torch.cuda.current_device()

    NC = 3
    DSOURCE = 128
    NBATCH = 64
    IMGSIZE = 64

    crop_size = 108
    re_size = 64
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(crop),
         transforms.ToPILImage(),
         transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])


    # wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
    # Make sure to change to correct path!
    imagenet_data = datasets.ImageFolder(args.dataset_dir,
                                         transform=transform)
    train_loader = torch.utils.data.DataLoader(imagenet_data,
                                               batch_size=NBATCH,
                                               shuffle=True,
                                               num_workers=0)

    g = GoodGenerator().to(device)
    phi = GoodDiscriminator().to(device)

    g_optim = torch.optim.Adam(g.parameters(), lr=1e-5, betas=(0.5, .999))
    # phi_optim = torch.optim.Adam(phi.parameters(), lr = 1e-4, betas=(0.5, .999))
    phi_optim = torch.optim.RMSprop(phi.parameters(), lr=1e-5)

    source = lambda N: torch.randn((N, DSOURCE)).type(dtype)

    # Fix latent samples for visualization purposes
    source_samples_plot = source(5 * 5)

    N_epochs = 10000
    N_critic = 1

    epsilon = 1

    def save_images(data_fake: torch.Tensor,
                    images_path: str, steps: int, epochs: int):
        fig = plt.figure(figsize=(5, 5))
        samples_plot = g(source_samples_plot).cpu().detach()
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

        # plt.show()

    def _c_epsilon_transform(y, C):
        return c_e_transform(y, C, epsilon=epsilon)

    def _objective(label_real, label_fake, C):
        return objective(label_fake, label_real, C, epsilon=epsilon)

    train_regularized_ot_GAN(g, phi, g_optim, phi_optim, DSOURCE,
                             _c_epsilon_transform, _objective,
                             save_images, train_loader,
                             args.output_dir,
                             max_epochs=N_epochs, max_steps=3000,
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
    parser.add_argument('--type', required=True, choices=['mnist', 'celeba'],
                        help='Selects which dataset/architecture '
                             'combination to use.')

    args = parser.parse_args()

    if args.type == 'mnist':
        train_mnist(args)
    else:
        train_celeba(args)


if __name__ == "__main__":
    main();
