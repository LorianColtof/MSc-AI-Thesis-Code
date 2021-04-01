import torch
from torch import nn
import torch.nn.functional as F


class Cifar10CNNGenerator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()

        channels = output_dim // (32 ** 2)

        self.initial_fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 1024),
            nn.BatchNorm1d(4 * 4 * 1024),
            nn.ReLU()
        )

        self.convolutions = nn.Sequential(
            # 4 x 4 x 1024
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 8 x 8 x 512
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 16 x 16 x 256
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 32 x 32 x 128
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 32 x 32 x 128
            nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1),

            # 32 x 32 x C
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.initial_fc(input)
        output = output.reshape(-1, 1024, 4, 4)
        output = self.convolutions(output)

        return output


class Cifar10CNNDiscriminator(nn.Module):
    def __init__(self, input_dim, include_final_linear=True,
                 final_linear_bias=True):
        super().__init__()

        channels = input_dim // (32 ** 2)

        self.model = nn.Sequential(
            # 32 x 32 x C
            nn.Conv2d(channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32 x 32 x 32
            nn.MaxPool2d(3, 2, 1),

            # 16 x 16 x 32
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 16 x 16 x 64
            nn.MaxPool2d(3, 2, 1),

            # 8 x 8 x 64
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 8 x 8 x 64
            nn.MaxPool2d(3, 2, 1),

            # 4 x 4 x 64
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 4 x 4 x 64
            nn.MaxPool2d(3, 2, 1),

            # 2 x 2 x 64
        )

        self.include_final_linear = include_final_linear

        if include_final_linear:
            self.final_linear = nn.Linear(256, 1, bias=final_linear_bias)

        self.normalize_final_linear()

    def forward(self, img):
        out = self.model(img.reshape(-1, 3, 32, 32))

        if self.include_final_linear:
            return self.final_linear(out.reshape(-1, 256))
        else:
            return out.reshape(-1, 256)

    def normalize_final_linear(self):
        if self.include_final_linear:
            self.final_linear.weight.data = F.normalize(
                self.final_linear.weight.data, p=2, dim=1)


class Cifar10DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()

        self.latent_dim = latent_dim

        self.preprocess = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 4 * latent_dim),
            nn.BatchNorm1d(4 * 4 * 4 * latent_dim),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * latent_dim, 2 * latent_dim, 2, stride=2),
            nn.BatchNorm2d(2 * latent_dim),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * latent_dim, latent_dim, 2, stride=2),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(latent_dim, 3, 2, stride=2)

        self.main = nn.Sequential(
            block1,
            block2,
            deconv_out,
            nn.Tanh()
        )

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.latent_dim, 4, 4)
        output = self.main(output)
        return output.view(-1, 3, 32, 32)


class Cifar10DCGANDiscriminator(nn.Module):
    def __init__(self, input_dim, include_final_linear=True,
                 final_linear_bias=True):
        super().__init__()

        dim = 128
        self.main = nn.Sequential(
            nn.Conv2d(3, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.include_final_linear = include_final_linear

        if include_final_linear:
            self.final_linear = nn.Linear(4 * 4 * 4 * dim, 1, bias=final_linear_bias)

        self.normalize_final_linear()

    def forward(self, img):
        output = self.main(img.reshape(-1, 3, 32, 32))
        output = output.view(-1, 4 * 4 * 4 * 128)

        if self.include_final_linear:
            output = self.final_linear(output)

        return output

    def normalize_final_linear(self):
        if self.include_final_linear:
            self.final_linear.weight.data = F.normalize(
                self.final_linear.weight.data, p=2, dim=1)
