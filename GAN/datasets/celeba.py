import os
from typing import List

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets
from torchvision.transforms import transforms

from configuration import Dataset
from datasets.base import AbstractBaseDataset


class CelebaDataset(AbstractBaseDataset):
    _source_samples_plot: torch.Tensor
    _plot_size = (5, 5)

    def __init__(self, dataset_config: Dataset, num_workers: int,
                 device: torch.device,
                 batch_size: int, latent_dimension: int,
                 drop_last: bool = False):
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
            imagenet_data, batch_size=batch_size, shuffle=True,
            pin_memory=True, drop_last=drop_last, num_workers=num_workers)

        # Fix latent samples for visualization purposes
        self._source_samples_plot = torch.randn(
            (self._plot_size[0] * self._plot_size[1],
             latent_dimension), device=device)

    def _imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        img = img.permute(1, 2, 0)
        plt.imshow(img)

    def _create_plot(self, img_path: str, samples: torch.Tensor):
        NC = 3
        IMGSIZE = 64

        fig = plt.figure(figsize=self._plot_size)

        for k in range(self._plot_size[0] * self._plot_size[1]):
            plt.subplot(self._plot_size[0], self._plot_size[1], k + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            self._imshow(samples[k].reshape(NC, IMGSIZE, IMGSIZE))
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0,
                            top=1)

        plt.savefig(img_path, dpi=75)
        plt.close(fig)

    def save_generated_data(self, generator_network: torch.nn.Module,
                            images_path: str, filename: str) -> List[str]:
        samples = generator_network(
            self._source_samples_plot).cpu().detach()
        img_path = os.path.join(images_path, f'{filename}.png')

        self._create_plot(img_path, samples)

        return [img_path]

    def save_real_data(self, images_path: str, filename: str) -> List[str]:
        samples = next(iter(self.dataloader))[0]
        img_path = os.path.join(images_path, f'{filename}.png')

        self._create_plot(img_path, samples)

        return [img_path]
