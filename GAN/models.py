import math
from typing import  Any

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 64
OUTPUT_DIM = 64  # 64*64*3


class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True,
                 stride=1, bias=True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size,
                              stride=stride, padding=self.padding, bias=bias)

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size,
                              he_init=self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:, :, ::2, ::2] +
                  output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] +
                  output[ :, :, 1::2, 1::2]) / 4
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size,
                              he_init=self.he_init)

    def forward(self, input):
        output = input
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] +
                  output[ :, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width,
                             self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [
            t_t.reshape(batch_size, input_height, output_width, output_depth)
            for t_t in spl]
        output = torch.stack(stacks, 0).transpose(0, 1).permute(0, 2, 1, 3,
                                                                4).reshape(
            batch_size, output_height, output_width, output_depth)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True,
                 bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size,
                              he_init=self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None,
                 hw=DIM):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            # TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim,
                                              kernel_size=1, he_init=False)
            self.conv_1 = MyConvo2d(input_dim, input_dim,
                                    kernel_size=kernel_size, bias=False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim,
                                       kernel_size=kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim,
                                              kernel_size=1, he_init=False)
            self.conv_1 = UpSampleConv(input_dim, output_dim,
                                       kernel_size=kernel_size, bias=False)
            self.conv_2 = MyConvo2d(output_dim, output_dim,
                                    kernel_size=kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim,
                                           kernel_size=1, he_init=False)
            self.conv_1 = MyConvo2d(input_dim, input_dim,
                                    kernel_size=kernel_size, bias=False)
            self.conv_2 = MyConvo2d(input_dim, output_dim,
                                    kernel_size=kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output


class ReLULayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(ReLULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.linear(input)
        output = self.relu(output)
        return output


class FCGenerator(nn.Module):
    def __init__(self, FC_DIM=512):
        super(FCGenerator, self).__init__()
        self.relulayer1 = ReLULayer(128, FC_DIM)
        self.relulayer2 = ReLULayer(FC_DIM, FC_DIM)
        self.relulayer3 = ReLULayer(FC_DIM, FC_DIM)
        self.relulayer4 = ReLULayer(FC_DIM, FC_DIM)
        self.linear = nn.Linear(FC_DIM, OUTPUT_DIM)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.relulayer1(input)
        output = self.relulayer2(output)
        output = self.relulayer3(output)
        output = self.relulayer4(output)
        output = self.linear(output)
        output = self.tanh(output)
        return output


class GoodGenerator(nn.Module):
    def __init__(self, latent_dim=DIM, output_dim=OUTPUT_DIM):
        super(GoodGenerator, self).__init__()

        output_dim_sqrt = int(math.sqrt(output_dim))
        if output_dim_sqrt ** 2 != output_dim:
            raise Exception("output_dim is not a square")

        self.dim = latent_dim
        self.output_dim = output_dim_sqrt

        self.ln1 = nn.Linear(128, 4 * 4 * 8 * self.dim)
        self.rb1 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(8 * self.dim, 4 * self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(4 * self.dim, 2 * self.dim, 3, resample='up')
        self.rb4 = ResidualBlock(2 * self.dim, 1 * self.dim, 3, resample='up')
        self.bn = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1 * self.dim, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln1(input.contiguous())
        output = output.view(-1, 8 * self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        output = output.view(-1, 3, self.output_dim, self.output_dim)
        return output


class GoodDiscriminator(nn.Module):
    def __init__(self, input_dim=DIM):
        super(GoodDiscriminator, self).__init__()

        input_dim_sqrt = int(math.sqrt(input_dim))
        if input_dim_sqrt ** 2 != input_dim:
            raise Exception("output_dim is not a square")

        self.dim = input_dim_sqrt

        self.conv1 = MyConvo2d(3, self.dim, 3, he_init=False)
        self.rb1 = ResidualBlock(self.dim, 2 * self.dim, 3, resample='down',
                                 hw=DIM)
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3,
                                 resample='down', hw=int(DIM / 2))
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3,
                                 resample='down', hw=int(DIM / 4))
        self.rb4 = ResidualBlock(8 * self.dim, 8 * self.dim, 3,
                                 resample='down', hw=int(DIM / 8))
        self.ln1 = nn.Linear(4 * 4 * 8 * self.dim, 1)

    def forward(self, input):
        output = input.contiguous()
        output = output.view(-1, 3, DIM, DIM)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, 4 * 4 * 8 * self.dim)
        output = self.ln1(output)
        return output


class WorseDiscriminator(nn.Module):
    def __init__(self, dim=64):
        super(WorseDiscriminator, self).__init__()
        self.dim = dim

        main = nn.Sequential(
            nn.Conv2d(3, self.dim, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(dim, 2 * self.dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(2 * self.dim, 4 * self.dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(4 * self.dim, 8 * self.dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4 * 4 * 8 * self.dim, 1)

    def forward(self, input):
        input = input.view(-1, 3, 64, 64)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 8 * self.dim)
        out = self.output(out)
        return out


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
    def __init__(self, input_dim, final_linear_bias=True):
        super().__init__()

        self.model = nn.Sequential(
            # 28 x 28 x 1
            nn.Conv2d(1, 32, 3, 1, 1),
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

        self.final_linear = nn.Linear(256, 1, bias=final_linear_bias)

        # self.normalize_final_linear()

    def forward(self, img):
        out = self.model(img)
        # return discriminator score for img
        return self.final_linear(out.reshape(-1, 256))

    def normalize_final_linear(self):
        self.final_linear.weight.data = F.normalize(
            self.final_linear.weight.data, p=2, dim=1)


_models = {
    'MnistGenerator': MnistGenerator,
    'MnistDiscriminator': MnistDiscriminator,
    'MnistCNNDiscriminator': MnistCNNDiscriminator,
    'GoodGenerator': GoodGenerator,
    'GoodDiscriminator': GoodDiscriminator
}


def load_model(model_type: str, **kwargs: Any) -> nn.Module:
    try:
        model = _models[model_type]
    except KeyError:
        raise Exception(f"Model '{model_type}' does not exist")

    return model(**kwargs)
