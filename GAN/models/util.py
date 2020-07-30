from torch import nn


class Conv2dResidualBlock(nn.Module):
    """
    Residual Block with instance normalization.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True,
                              track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)
