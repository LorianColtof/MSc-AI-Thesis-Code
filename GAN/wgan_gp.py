import argparse
import os
import torch
import torch.nn as nn
import torch.cuda
import torch.optim
import torch.autograd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 784
        #   Output non-linearity

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        # return discriminator score for img
        return self.model(img)


def gradient_penalty(discriminator, samples_real, samples_generated,
                     lambda_reg=5):
    device = samples_real.device

    batch_size = samples_real.shape[0]
    alpha = torch.rand(batch_size, 1, device=device)\
        .expand(samples_real.size())

    interpolates = alpha * samples_real + ((1 - alpha) * samples_generated)

    interpolates_var = torch.autograd.Variable(
        interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates_var)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates_var,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        * lambda_reg)
    return gradient_penalty


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D,
          device):

    def data_iterator():
        while True:
            for data in dataloader:
                yield data

    def get_imgs():
        imgs_cpu, _ = next(data_it)
        batch_size = imgs_cpu.size(0)
        imgs_cuda = imgs_cpu.to(device).reshape(batch_size, -1)

        z = torch.randn((batch_size, args.latent_dim), device=device)
        imgs_generated = generator(z)

        return imgs_cuda, imgs_generated

    data_it = data_iterator()

    for step in range(args.n_steps):
        print(f"Step {step}")

        # === Train discriminator ===

        for _ in range(args.n_discriminator):
            imgs, generated_samples = get_imgs()

            penalty = gradient_penalty(discriminator, imgs, generated_samples)
            disc_generated = discriminator(generated_samples)
            disc_real = -discriminator(imgs)
            loss_discriminator = (-(disc_generated.mean() + disc_real.mean()) +
                                  penalty)

            optimizer_D.zero_grad()
            loss_discriminator.backward()
            optimizer_D.step()

        # === Train generator ===

        imgs, generated_samples = get_imgs()

        disc_generated = discriminator(generated_samples)
        disc_real = -discriminator(imgs)
        loss_generator = disc_generated.mean() + disc_real.mean()

        optimizer_G.zero_grad()
        loss_generator.backward()
        optimizer_G.step()

        # Save Images
        # -----------
        # batches_done = epoch * len(dataloader) + i
        if step % args.save_interval == 0:
            print("Saving images")

            # You can use the function save_image(Tensor (shape Bx1x28x28),
            # filename, number of rows, normalize) to save the generated
            # images, e.g.:
            save_image(generated_samples[:25].reshape(-1, 1, 28, 28),
                       'images_wgan_gp/step_{}.png'.format(
                           step), nrow=5, normalize=True)


def main():
    # Create output image directory
    os.makedirs('images_wgan_gp', exist_ok=True)

    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device('cuda')
    else:
        print("CUDA is not available, falling back to CPU")
        device = torch.device('cpu')

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    img_dim = 28 * 28

    # Initialize models and optimizers
    generator = Generator(args.latent_dim, img_dim)
    discriminator = Discriminator(img_dim)

    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=args.lr_generator)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(),
                                      lr=args.lr_discriminator)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D,
          device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "gan.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=10000,
                        help='number of steps')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr_generator', type=float, default=0.0002,
                        help='learning rate of generator')
    parser.add_argument('--lr_discriminator', type=float, default=1e-4,
                        help='learning rate of discriminator')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--n_discriminator', type=int, default=5,
                        help='amount of iterations to train discriminator')
    parser.add_argument('--save_interval', type=int, default=25,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
