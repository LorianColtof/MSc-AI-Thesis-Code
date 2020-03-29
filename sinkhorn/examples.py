import torch
import torch.cuda
import torch.random
import matplotlib.pyplot as plt
import math
import time

from ot.smooth import smooth_ot_dual

from sinkhorn.regular import sinkhorn
from sinkhorn.quadratic import sinkhorn_quadratic_cyclic_projection,\
    sinkhorn_quadratic_gradient_descent,\
    sinkhorn_quadratic_fixed_point_iteration,\
    sinkhorn_quadratic_nesterov_gradient_descent
from sinkhorn.tsallis import sinkhorn_tsallis_entropy
from sinkhorn.unbalanced import sinkhorn_unbalanced


def euc_costs(n, scale, device='cpu', dtype=torch.float32):
    t = torch.linspace(0, 1 - 1.0 / n, n, device=device, dtype=dtype)
    x, y = torch.meshgrid([t, t])
    return (x - y)**2 * scale


def example_point_clouds(device: torch.device):
    N = [200, 300]

    x = torch.rand(2, N[0], device=device) - .5
    theta = 2 * math.pi * torch.rand(1, N[1], device=device)
    r = .8 + .2 * math.pi * torch.rand(1, N[1], device=device)
    y = torch.stack((torch.cos(theta) * r, torch.sin(theta) * r)).squeeze()

    plotp = lambda x, col: plt.scatter(x[0, :], x[1, :],
                                       s=200, edgecolors="k",
                                       c=col, linewidths=2)

    plt.figure(figsize=(10, 10))

    plotp(x.cpu(), 'b')
    plotp(y.cpu(), 'r')

    plt.axis("off")
    plt.xlim(torch.min(y[0, :]).cpu() - .1, torch.max(y[0, :]).cpu() + .1)
    plt.ylim(torch.min(y[1, :]).cpu() - .1, torch.max(y[1, :]).cpu() + .1)

    plt.show()

    x2 = torch.sum(x ** 2, 0)
    y2 = torch.sum(y ** 2, 0)

    C = x2.reshape(-1, 1) + y2.reshape(1, -1) + 2 * x.T @ y

    plt.imshow(C.cpu())
    plt.show()

    a = torch.ones(N[0], device=device) / N[0]
    b = torch.ones(N[1], device=device) / N[1]
    # epsilon = .01
    epsilon = .1

    start = time.time()

    def reg_entropy_convex_conjugate(x):
        return torch.exp(x)

    def reg_entropy_derivative_inverse(x):
        return torch.exp(-x)

    P = sinkhorn(C, a, b, epsilon, log=True)

    elapsed = time.time() - start

    print(f"Elapsed time: {elapsed} seconds")

    plt.imshow(P.cpu())
    plt.show()


def example_gaussians(device: torch.device):
    sigma = .06

    n = 200
    t = torch.linspace(0, 1, n, device=device, dtype=torch.float64)
    Gaussian = lambda t0, sigma: torch.exp(-(t - t0) ** 2 / (2 * sigma ** 2))
    normalize = lambda p: p / p.sum()

    a = Gaussian(.25, sigma)
    b = Gaussian(.8, sigma)

    vmin = .02
    a = normalize(a + a.max() * vmin)
    b = normalize(b + b.max() * vmin)

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.bar(t.cpu(), a.cpu(), width=1 / len(t), color="darkblue")
    plt.subplot(2, 1, 2)
    plt.bar(t.cpu(), b.cpu(), width=1 / len(t), color="darkblue")

    plt.show()

    # Some of the algorithms are sensitive to which one we use
    X, Y = torch.meshgrid([t, t])
    C = (X - Y) ** 2
    # C = euc_costs(n, n, device, dtype=t.dtype)

    # epsilon = 9e-4
    epsilon = 1e-2
    # epsilon = 1e-1

    def reg_entropy_convex_conjugate(x):
        return torch.exp(x)

    def reg_entropy_derivative_inverse(x):
        return torch.exp(-x)

    def reg_quadratic_convex_conjugate(x):
        return x**2 / 2

    def reg_quadratic_derivative_inverse(x):
        return x

    start = time.time()

    log = True
    gamma = 1.0 / epsilon

    P_regular = sinkhorn(C, a, b, epsilon)
    P_unbalanced = sinkhorn_unbalanced(C, a, b, epsilon, 1, log=log)

    plt.imshow(torch.log(P_regular + 1e-5).cpu())
    plt.title('Normal Sinkhorn')
    plt.show()

    plt.imshow(torch.log(P_unbalanced + 1e-5).cpu())
    plt.title('Unbalanced Sinkhorn')
    plt.show()

    P_cp = sinkhorn_quadratic_cyclic_projection(C, a, b, gamma, log=log)
    print("Cyclic projection done")

    P_gd = sinkhorn_quadratic_gradient_descent(C, a, b, gamma, log=log)
    print("Gradient descent done")

    P_fpi = sinkhorn_quadratic_fixed_point_iteration(C, a, b, gamma, log=log)
    print("Fixed point iteration done")

    P_ngd = sinkhorn_quadratic_nesterov_gradient_descent(C, a, b, gamma,
                                                         log=log)
    print("Nesterov gradient descent done")

    P_reference = torch.from_numpy(smooth_ot_dual(
        a.cpu().numpy(), b.cpu().numpy(), C.cpu().numpy(), gamma))

    elapsed = time.time() - start

    print(f"Elapsed time: {elapsed} seconds")

    for P, title in [(P_cp, "Cyclic projection"),
                     (P_gd, "Gradient descent"),
                     (P_fpi, "Fixed point iteration"),
                     (P_ngd, "Nesterov gradient descent"),
                     (P_reference, "POT reference")]:
        plt.imshow(torch.log(P + 1e-5).cpu())
        plt.title(title)
        plt.show()


def main():
    torch.random.manual_seed(42)

    use_cpu = True

    if torch.cuda.is_available() and not use_cpu:
        print("Using CUDA")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    # example_point_clouds(device)
    example_gaussians(device)

if __name__ == "__main__":
    main()
