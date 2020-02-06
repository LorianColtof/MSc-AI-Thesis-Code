import torch
import torch.cuda
import torch.random
import matplotlib.pyplot as plt
import math

import time


def sinkhorn(C: torch.tensor, a: torch.tensor, b: torch.tensor,
             epsilon: float, num_iter: int = 1000,
             converge_error: float = 1e-8):
    K = torch.exp(-C / epsilon)
    v = torch.ones_like(b)
    u = torch.ones_like(a)

    for i in range(num_iter):
        print(f"Iteration {i}")
        u_prev = u
        u = a / (K @ v)
        v = b / (K.T @ u)

        u_diff = (u_prev - u).abs().sum()
        print(f"u_diff={u_diff}")
        if u_diff < converge_error:
            break

    return torch.diag(u) @ K @ torch.diag(v)


def example_point_clouds(device: torch.device):
    N = [300, 200]

    x = torch.rand(2, N[0], device=device) - .5
    theta = 2 * math.pi * torch.rand(1, N[1], device=device)
    r = .8 + .2 * math.pi * torch.rand(1, N[1], device=device)
    y = torch.stack((torch.cos(theta) * r, torch.sin(theta) * r)).squeeze()

    plotp = lambda x, col: plt.scatter(x[0, :], x[1, :],
                                       s=200, edgecolors="k",
                                       c=col, linewidths=2)

    plt.figure(figsize=(10, 10))

    # plotp(x.cpu(), 'b')
    # plotp(y.cpu(), 'r')

    # plt.axis("off")
    # plt.xlim(torch.min(y[0, :]).cpu() - .1, torch.max(y[0, :]).cpu() + .1)
    # plt.ylim(torch.min(y[1, :]).cpu() - .1, torch.max(y[1, :]).cpu() + .1)
    #
    # plt.show()

    x2 = torch.sum(x**2, 0)
    y2 = torch.sum(y**2, 0)

    C = x2.reshape(-1, 1) + y2.reshape(1, -1) + 2 * x.T @ y

    # plt.imshow(C.cpu())
    # plt.show()

    a = torch.ones(N[0], device=device) / N[0]
    b = torch.ones(N[1], device=device) / N[1]
    # epsilon = .01
    epsilon = .1

    start = time.time()
    # P = sinkhorn_log(C, a, b, epsilon)
    P = sinkhorn(C, a, b, epsilon)
    elapsed = time.time() - start

    print(f"Elapsed time: {elapsed} seconds")

    plt.imshow(P.cpu())
    plt.show()


def example_gaussians(device: torch.device):
    sigma = .06

    t = torch.linspace(0, 1, 200, device=device) #, dtype=torch.float64)
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

    X, Y = torch.meshgrid([t, t])
    C = (X - Y)**2

    # epsilon = 9e-4
    epsilon = 1e-2

    start = time.time()
    P = sinkhorn(C, a, b, epsilon, num_iter=5000)
    elapsed = time.time() - start

    print(f"Elapsed time: {elapsed} seconds")

    plt.imshow(torch.log(P + 1e-5).cpu())
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

    example_point_clouds(device)
    example_gaussians(device)


if __name__ == "__main__":
    main()