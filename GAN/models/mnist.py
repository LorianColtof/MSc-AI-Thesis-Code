from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.util import Conv2dResidualBlock


class MnistGenerator(nn.Module):
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


class MnistCNNGenerator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()

        channels = output_dim // (28 ** 2)

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
            nn.Conv2d(128, channels, kernel_size=5, stride=1, padding=0),

            # 28 x 28 x C
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.initial_fc(input)
        output = output.reshape(-1, 1024, 4, 4)
        output = self.convolutions(output)

        return output


class MnistCNNEnhancedGenerator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int,
                 res_block_repeat: int = 3):
        super().__init__()

        channels = output_dim // (28 ** 2)

        self.initial_fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 1024),
            nn.BatchNorm1d(4 * 4 * 1024),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(*[
            Conv2dResidualBlock(dim_in=1024, dim_out=1024)
            for _ in range(res_block_repeat)
        ])

        self.convolutions = nn.Sequential(
            # 4 x 4 x 1024
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.InstanceNorm2d(512, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # 8 x 8 x 512
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # 16 x 16 x 256
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # 32 x 32 x 128
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2,
                      bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # 32 x 32 x 128
            nn.Conv2d(128, channels, kernel_size=5,
                      stride=1, padding=0, bias=False),

            # 28 x 28 x C
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.initial_fc(input)
        output = output.reshape(-1, 1024, 4, 4)
        output = self.res_blocks(output)
        output = self.convolutions(output)

        return output


class MnistDiscriminator(nn.Module):
    def __init__(self, input_dim, final_linear_bias=True):
        super().__init__()

        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )

        self.final_linear = nn.Linear(64, 1, bias=final_linear_bias)

        # self.normalize_final_linear()

    def forward(self, img):
        img = img.reshape(-1, self.input_dim)

        out = self.model(img)
        # return discriminator score for img
        return self.final_linear(out)

    def normalize_final_linear(self):
        self.final_linear.weight.data = F.normalize(
            self.final_linear.weight.data, p=2, dim=1)


class MnistCNNDiscriminator(nn.Module):
    def __init__(self, input_dim, include_final_linear=True,
                 final_linear_bias=True):
        super().__init__()

        channels = input_dim // (28 ** 2)

        self.model = nn.Sequential(
            # 28 x 28 x C
            nn.Conv2d(channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 28 x 28 x 32
            nn.MaxPool2d(3, 2, 1),

            # 14 x 14 x 32
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 14 x 14 x 64
            nn.MaxPool2d(3, 2, 1),

            # 7 x 7 x 64
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 7 x 7 x 64
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
        out = self.model(img)

        if self.include_final_linear:
            return self.final_linear(out.reshape(-1, 256))
        else:
            return out.reshape(-1, 256)

    def normalize_final_linear(self):
        if self.include_final_linear:
            self.final_linear.weight.data = F.normalize(
                self.final_linear.weight.data, p=2, dim=1)


class MnistCNNEnhancedDiscriminatorWithClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.main = MnistCNNEnhancedDiscriminator(
            input_dim, include_final_linear=False)
        self.conv_cls = nn.Conv2d(64, num_classes, kernel_size=2, bias=False)

    def forward(self, input: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        out_src = self.main(input).reshape(-1, 64, 2, 2)
        out_cls = self.conv_cls(out_src)

        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class MnistCNNEnhancedDiscriminator(nn.Module):
    def __init__(self, input_dim, include_final_linear=True,
                 final_linear_bias=True):
        super().__init__()

        channels = input_dim // (28 ** 2)

        self.model = nn.Sequential(
            # 28 x 28 x C
            nn.Conv2d(channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),

            # 28 x 28 x 32
            nn.MaxPool2d(3, 2, 1),

            # 14 x 14 x 32
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            # 14 x 14 x 64
            nn.MaxPool2d(3, 2, 1),

            # 7 x 7 x 64
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            # 7 x 7 x 64
            nn.MaxPool2d(3, 2, 1),

            # 4 x 4 x 64
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            # 4 x 4 x 64
            nn.MaxPool2d(3, 2, 1),

            # 2 x 2 x 64
        )

        self.include_final_linear = include_final_linear

        if include_final_linear:
            self.final_linear = nn.Linear(256, 1, bias=final_linear_bias)

        self.normalize_final_linear()

    def forward(self, img):
        out = self.model(img)

        if self.include_final_linear:
            return self.final_linear(out.reshape(-1, 256))
        else:
            return out.reshape(-1, 256)

    def normalize_final_linear(self):
        if self.include_final_linear:
            self.final_linear.weight.data = F.normalize(
                self.final_linear.weight.data, p=2, dim=1)


class MnistEncoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int,
                 res_block_repeat: int = 3):
        super().__init__()

        channels = output_dim // (28 ** 2)

        self.convolutions = nn.Sequential(
            # 28 x 28 x C
            nn.Conv2d(channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),

            # 28 x 28 x 32
            nn.MaxPool2d(3, 2, 1),

            # 14 x 14 x 32
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            # 14 x 14 x 64
            nn.MaxPool2d(3, 2, 1),

            # 7 x 7 x 64
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            # 7 x 7 x 64
            nn.MaxPool2d(3, 2, 1),

            # 4 x 4 x 64
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            # 4 x 4 x 64
        )

        self.res_blocks = nn.Sequential(*[
            Conv2dResidualBlock(dim_in=64, dim_out=64)
            for _ in range(res_block_repeat)
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.convolutions(input)
        output = self.res_blocks(output)

        return output.reshape(-1, 4 * 4 * 64)
