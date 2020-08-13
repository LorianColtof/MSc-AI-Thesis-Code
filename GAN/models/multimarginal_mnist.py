import torch
from torch import nn

from models.util import Conv2dResidualBlock


class MultimarginalMnistEncoder(nn.Module):
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


class MultimarginalMnistEnhancedEncoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int,
                 res_block_repeat: int = 3):
        super().__init__()

        assert latent_dim == 8 ** 2 * 128, "Using wrong latent space dimension"

        channels = output_dim // (28 ** 2)

        self.convolutions = nn.Sequential(
            # 28 x 28 x C
            nn.Conv2d(channels, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # 28 x 28 x 32

            nn.Conv2d(32, 64, kernel_size=3,
                      stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # 16 x 16 x 64

            nn.Conv2d(64, 128, kernel_size=5,
                      stride=2, padding=2, bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # 8 x 8 x 128
        )

        self.res_blocks = nn.Sequential(*[
            Conv2dResidualBlock(dim_in=128, dim_out=128)
            for _ in range(res_block_repeat)
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.convolutions(input)
        output = self.res_blocks(output)

        return output.reshape(-1, 8 * 8 * 128)


class MultimarginalMnistDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, res_block_repeat: int = 3):
        super().__init__()

        assert latent_dim == 8 ** 2 * 128, "Using wrong latent space dimension"

        channels = output_dim // (28 ** 2)

        # 8 x 8 x 128

        self.res_blocks = nn.Sequential(*[
            Conv2dResidualBlock(dim_in=128, dim_out=128)
            for _ in range(res_block_repeat)
        ])

        self.convolutions = nn.Sequential(
            # 8 x 8 x 128

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # 16 x 16 x 64

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # 32 x 32 x 32

            nn.Conv2d(32, channels, kernel_size=5,
                      stride=1, padding=0, bias=False),
            nn.Tanh()

            # 28 x 28 x C
        )

    def forward(self, input: torch.Tensor):
        output = self.res_blocks(input.reshape(-1, 128, 8, 8))
        output = self.convolutions(output)
        return output


class MultimarginalMnistDiscriminatorWithClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        channels = input_dim // (28 ** 2)

        self.main = nn.Sequential(
            # 28 x 28 x C
            nn.Conv2d(channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.01),

            # 28 x 28 x 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),

            # 14 x 14 x 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),

            # 7 x 7 x 128
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),

            # 7 x 7 x 128
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),

            # 3 x 3 x 256
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),

            # 3 x 3 x 256
            nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.01),

            # 2 x 2 x 256
        )

        self.conv_src = nn.Conv2d(
            256, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # -> 2 x 2 x 1

        self.conv_cls = nn.Conv2d(
            256, num_classes, kernel_size=4, stride=1, padding=1, bias=False)
        # -> 1 x 1 x num_classes

    def forward(self, input: torch.Tensor):
        h = self.main(input)
        out_src = self.conv_src(h)
        out_cls = self.conv_cls(h)
        return out_src, out_cls.squeeze()