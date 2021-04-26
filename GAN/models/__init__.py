from typing import Any

from models.celeba import *
from models.mnist import *
from models.cifar10 import *
from models.lsun_bedrooms import *
from models.multimarginal_celeba import *
from models.multimarginal_mnist import *


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)


class IdentityDiscriminator(nn.Module):
    def __init__(self, input_dim, include_final_linear=True,
                 final_linear_bias=True):
        super().__init__()

        self.input_dim = input_dim
        self.include_final_linear = include_final_linear

        if include_final_linear:
            self.final_linear = nn.Linear(self.input_dim,
                                          1, bias=final_linear_bias)

        self.normalize_final_linear()

    def forward(self, img):
        out = img.squeeze()
        if self.include_final_linear:
            return self.final_linear(out.reshape(-1, self.input_dim))
        else:
            return out

    def normalize_final_linear(self):
        if self.include_final_linear:
            self.final_linear.weight.data = F.normalize(
                self.final_linear.weight.data, p=2, dim=1)


class IdentitySourceEncoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()

        assert latent_dim == output_dim, "Latent dimension should " \
                                         "be the same as the output dimension"

    def forward(self, x):
        return x.squeeze()


class SimpleMLPGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z):
        return self.model(z)


class SimpleMLPDiscriminator(nn.Module):
    def __init__(self, input_dim, final_linear_bias=True):
        super().__init__()

        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1, bias=final_linear_bias)
        )

    def forward(self, data):
        return self.model(data.reshape(-1, self.input_dim))


class SimpleMLPDiscriminatorWithClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.input_dim = input_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.final_src = nn.Linear(32, 1, bias=False)
        self.final_cls = nn.Linear(32, num_classes, bias=False)

    def forward(self, data):
        h = self.main(data.reshape(-1, self.input_dim))
        out_src = self.final_src(h)
        out_cls = self.final_cls(h)

        return out_src, out_cls.squeeze()


class SimpleMLPCostFunction(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim

        layers = [
            nn.Linear(input_dim, 64),
            # nn.LeakyReLU(0.2),
            # nn.Linear(64, 64),
            # CReLU(),
            # nn.Linear(input_dim, input_dim),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

        for l in layers:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)

    def forward(self, data):
        return self.model(data.reshape(-1, self.input_dim))


def load_model(model_type: str, **kwargs: Any) -> nn.Module:
    try:
        model = globals()[model_type]
    except KeyError:
        raise Exception(f"Model '{model_type}' does not exist")

    return model(**kwargs)
