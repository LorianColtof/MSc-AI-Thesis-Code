import argparse
import os
import torch
import torch.nn as nn
import torch.cuda
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

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.model(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D,
          device):

    loss_fn = nn.BCELoss()

    for epoch in range(args.n_epochs):
        print("========== Epoch: {} ==========".format(epoch))

        losses_G = []
        losses_D = []

        for i, (imgs_cpu, _) in enumerate(dataloader):

            imgs = imgs_cpu.to(device)

            batch_size = imgs.size(0)

            z = torch.randn((batch_size, args.latent_dim), device=device)
            generated_samples = generator(z)

            ones = torch.FloatTensor(batch_size, 1).uniform_(0.7, 1.2)\
                .to(device)
            zeros = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.3)\
                .to(device)

            discriminator_output_real = discriminator(
                imgs.reshape(batch_size, -1))
            discriminator_output_fake = discriminator(generated_samples)

            loss_discriminator = (loss_fn(discriminator_output_real, ones) +
                                  loss_fn(discriminator_output_fake, zeros))
            loss_generator = loss_fn(discriminator_output_fake, ones)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            loss_generator.backward(retain_graph=True)
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            loss_discriminator.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print("{: >3}:    Loss D: {:.3f} loss G: {:.3f}".format(
                    i, loss_discriminator, loss_generator))

            losses_G.append(loss_generator.item())
            losses_D.append(loss_discriminator.item())

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:

                print("Saving images")
                print("Average loss D: {:.3f}  Average loss G: {:.3f}".format(
                    np.mean(losses_D), np.mean(losses_G)))

                losses_D = []
                losses_G = []

                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(generated_samples[:25].reshape(-1, 1, 28, 28),
                           'images_gan/epoch_{}_{}.png'.format(
                               epoch, batches_done), nrow=5, normalize=True)


def main():
    # Create output image directory
    os.makedirs('images_gan', exist_ok=True)

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

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D,
          device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "gan.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
