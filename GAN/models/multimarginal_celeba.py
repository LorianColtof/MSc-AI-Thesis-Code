import torch.nn as nn

from models.util import Conv2dResidualBlock


class MWGANCelebaDiscriminator(nn.Module):
    """
    Discriminator network with PatchGAN.
    """
    def __init__(self, input_dim, num_classes,
                 image_size=128, conv_dim=64, repeat_num=6):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim,
                                kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        # 64 x 64 x 64

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2,
                                    kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        # 2 x 2 x 256

        kernel_size = int(image_size / (2 ** repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # -> 2 x 2 x 1

        self.conv2 = nn.Conv2d(curr_dim, num_classes,
                               kernel_size=kernel_size, bias=False)
        # -> 1 x 1 x num_classes

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class MWGANCelebaResEncoder(nn.Module):
    """
    Encoder network.
    """
    def __init__(self, latent_dim, output_dim, conv_dim=64, repeat_num=3):
        super().__init__()

        layers = []

        # 3 x 128 x 128
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True,
                                        track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # 64 x 128 x 128

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True,
                                            track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # 128 x 64 x 64
        # 256 x 32 x 32

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(Conv2dResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out


class MWGANCelebaResDecoder(nn.Module):
    """
    Decoder network.
    """
    def __init__(self, latent_dim, output_dim, conv_dim=64, repeat_num=3):
        super().__init__()

        layers = []
        # downsampling 2^2
        curr_dim = conv_dim * 4
        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(Conv2dResidualBlock(dim_in=curr_dim,
                                              dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2,
                                             kernel_size=4, stride=2,
                                             padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True,
                                            track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, h):
        out = self.main(h)
        return out
