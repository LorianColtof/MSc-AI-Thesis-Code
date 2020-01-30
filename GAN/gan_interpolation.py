import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from gan import Generator


@torch.no_grad()
def main():
    generator = Generator(100, 28 * 28)
    generator.load_state_dict(torch.load('gan.pt'))
    generator.eval()

    n_steps = 9
    for n in range(50):
        z1 = torch.randn(100) * 1.5
        z2 = torch.randn(100) * 1.5

        print(n)
        print(z1)
        print(z2)
        print()

        step = (z2 - z1) / (n_steps - 1)

        imgs = []
        for i in range(n_steps):
            z = z1 + i * step

            img = generator(z.reshape(1, -1))

            imgs.append(img.reshape(28, 28))

        img_tensor = torch.stack(imgs).reshape(-1, 1, 28, 28)

        save_image(img_tensor,
                   'images_gan/gan_interpolation_{}.png'.format(n),
                   nrow=n_steps)

if __name__ == "__main__":
    main()
