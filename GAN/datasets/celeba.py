import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL.Image import Image
from torchvision import datasets
from torchvision.transforms import transforms

from configuration import Dataset
from datasets.base import AbstractBaseDataset


class CelebaDataset(AbstractBaseDataset):
    _source_samples_plot: torch.Tensor

    def __init__(self, dataset_config: Dataset, device: torch.device,
                 batch_size: int, latent_dimension: int):
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
             transforms.Resize(size=(re_size, re_size),
                               interpolation=Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

        self.data_dimension = re_size ** 2

        # wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets
        # /celeba.zip
        # Make sure to change to correct path!
        imagenet_data = datasets.ImageFolder(dataset_config.directory,
                                             transform=transform)
        self.dataloader = torch.utils.data.DataLoader(
            imagenet_data, batch_size=batch_size,
            shuffle=True, num_workers=4)

        # Fix latent samples for visualization purposes
        self._source_samples_plot = torch.randn(
            (5 * 5, latent_dimension), device=device)

    def _imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def save_generated_data(self, generator_network: torch.nn.Module,
                            images_path: str, steps: int, epochs: int) -> None:
        NC = 3
        IMGSIZE = 64

        fig = plt.figure(figsize=(5, 5))
        samples_plot = generator_network(
            self._source_samples_plot).cpu().detach()

        for k in range(5 * 5):
            plt.subplot(5, 5, k + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            self._imshow(samples_plot[k].reshape(NC, IMGSIZE, IMGSIZE))
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0,
                            top=1)

        plt.savefig(os.path.join(
            images_path, 'epoch_{}_step_{}.png'.format(epochs, steps)), dpi=75)
        plt.close(fig)


