from typing import Any

from models.celeba import *
from models.mnist import *
from models.mwgan import *


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


def load_model(model_type: str, **kwargs: Any) -> nn.Module:
    try:
        model = globals()[model_type]
    except KeyError:
        raise Exception(f"Model '{model_type}' does not exist")

    return model(**kwargs)
