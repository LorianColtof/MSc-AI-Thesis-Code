import os

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd

import PIL.Image as Image

import torchvision
from torchvision import datasets, transforms

from torch.autograd import grad
import functools

torch.backends.cudnn.enabled = False
dtype = torch.cuda.FloatTensor
device = torch.cuda.current_device()
print(torch.cuda.get_device_name(device))

# dtype = torch.FloatTensor
# device = 'cpu'

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
imagenet_data = datasets.ImageFolder('/home/lorian/Thesis-data/',
                                     transform=transform)
train_loader = torch.utils.data.DataLoader(imagenet_data,
                                           batch_size=NBATCH,
                                           shuffle=True,
                                           num_workers=4)



DIM = 64
OUTPUT_DIM = 64  # 64*64*3


class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True,
                 stride=1, bias=True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1,
                              padding=self.padding, bias=bias)

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
        output = output.permute(0, 3, 1, 2)
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
    def __init__(self, dim=DIM, output_dim=OUTPUT_DIM):
        super(GoodGenerator, self).__init__()

        self.dim = dim
        self.output_dim = output_dim

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
    def __init__(self, dim=DIM):
        super(GoodDiscriminator, self).__init__()

        self.dim = dim

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


def objective(y0, y1, C, epsilon=1):
    val0 = torch.mean(y0)
    val1 = torch.mean(y1)

    tmp0 = (-C + (y0 + torch.t(y1)))
    val_reg = epsilon * torch.mean(torch.exp(tmp0 / epsilon))

    val = val0 + val1 - val_reg
    return val


def c_transform(y, C, epsilon=1):
    return softmin((C - y), epsilon=epsilon)[:None]


def softmin(X, epsilon=1):
    Y = -X / epsilon
    Ymax = torch.max(Y, 0)[0][:, None]
    return -epsilon * (
                Ymax + torch.log(torch.mean(torch.exp(Y - Ymax.t()), 0))[:,
                       None])


def sinkhorn(a, b, C, epsilon=1):
    n = len(a)
    m = len(b)

    u0 = (torch.ones((n, 1)) / n).type(dtype)
    v0 = (torch.ones((m, 1)) / m).type(dtype)
    K = torch.exp(-epsilon * C)

    gamma = u0 * K * (v0.t())
    gamma = gamma / torch.sum(gamma)

    k = 0
    while (k < 5):
        k += 1
        u1 = a / (K @ v0)
        v1 = b / (K.t() @ u1)

        u0 = u1
        v0 = v1

        gamma = u1 * K * (v1.t())

    return gamma  # np.sqrt(torch.sum(gamma*C)),gamma.view(-1)


g = GoodGenerator().to(device)
phi = GoodDiscriminator().to(device)

g_optim = torch.optim.Adam(g.parameters(), lr=1e-5, betas=(0.5, .999))
# phi_optim = torch.optim.Adam(phi.parameters(), lr = 1e-4, betas=(0.5, .999))
phi_optim = torch.optim.RMSprop(phi.parameters(), lr=1e-5)

source = lambda N: torch.randn((N, DSOURCE)).type(dtype)

N_epochs = 10000
N_critic = 1

p = 2
q = 2

weights = (torch.ones((NBATCH, 1)) / NBATCH).type(dtype)

# cost = lambda x, y: dist_pq(x, y, p=p, q=q)

epsilon = 1

w_hist = []
k_iter = 0
k_img = 0
endit = False

# Fix latent samples for visualization purposes
source_samples_plot = source(5 * 5)

# Start training
for i in range(N_epochs):
    if endit == True:
        break
    for x_real, _ in train_loader:
        if x_real.shape[0] is not NBATCH:
            break
        x_real = x_real.type(dtype)

        phi_optim.zero_grad()
        g_optim.zero_grad()

        # x_real are real samples, now sample from the model (x_fake),
        # and compute the cost matrix C for the c-transform
        with torch.no_grad():
            samples = source(NBATCH)
            x_fake = g(samples)
            C = dist(x_fake, x_real, p=p, q=q)

        # Train the discriminator
        for j in range(N_critic):
            y_fake = phi(x_fake)
            y_real = c_transform(y_fake, C, epsilon=epsilon)

            loss = -objective(y_fake, y_real, C, epsilon=epsilon)

            print(loss)

            loss.backward()


            phi_optim.step()
            phi_optim.zero_grad()

        g_optim.zero_grad()

        # Train the generator
        x_fake = g(samples)

        C = dist(x_fake, x_real, p=p, q=q)

        y_fake = phi(x_fake)
        y_real = c_transform(y_fake, C, epsilon=epsilon)

        loss = objective(y_fake, y_real, C, epsilon=epsilon)
        print(loss)
        loss.backward()

        g_optim.step()
        g_optim.zero_grad()

        # print('%0.1f' % float(loss), end=', ')
        # if k_iter % 10 == 0 and k_iter > 0:
        #    print('\n')
        # w_hist.append(loss)

        if k_iter % 200 == 0 and k_iter > 0:
            k_img += 1
            fig = plt.figure(figsize=(5, 5))
            samples_plot = g(source_samples_plot).detach()
            for k in range(5 * 5):
                plt.subplot(5, 5, k + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                imshow(samples_plot[k].cpu().reshape(NC, IMGSIZE, IMGSIZE))
                plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0,
                                top=1)

            plt.savefig('images_sinkhorn_gan_augusto/fig%03d.png' % k_img, dpi=75)
            plt.close(fig)
            # plt.show()

            print('%dth Epoc done' % i)
            print('Iterations: ', k_iter)
            print('loss:', float(loss))
        k_iter += 1
        if k_iter > 300000:
            endit = True
