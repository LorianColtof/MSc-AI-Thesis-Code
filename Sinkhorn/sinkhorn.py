import torch
import torch.cuda
import torch.random
import matplotlib.pyplot as plt
import math

import time

from ot.smooth import smooth_ot_dual


def sinkhorn(C: torch.tensor, a: torch.tensor, b: torch.tensor,
             epsilon: float, num_iter: int = 1000,
             converge_error: float = 1e-8):
    K = torch.exp(-C / epsilon)
    v = torch.ones_like(b)
    u = torch.ones_like(a)

    def transform_C_f_g(f, g):
        return (C - f.expand_as(C.T).T - g.expand_as(C)) / epsilon

    def calculate_loss(f, g):
        return -(f @ a + g @ b
                 - epsilon * torch.exp(
                    transform_C_f_g(f, g)).sum())

    losses = []

    for i in range(num_iter):
        print(f"Iteration {i}")
        u_prev = u
        u = a / (K @ v)
        v = b / (K.T @ u)

        u_diff = (u_prev - u).abs().sum()
        print(f"u_diff={u_diff}")
        if u_diff < converge_error:
            break

        f = epsilon * u.log()
        g = epsilon * v.log()

        # print(f)
        # print(g)

        loss = -calculate_loss(f, g).item()
        losses.append(loss)

    # plt.plot(list(range(len(losses))), losses)
    # plt.show()

    return torch.diag(u) @ K @ torch.diag(v), f, g


def sinkhorn_quadratic_cyclic_projection(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon: float,
        num_iter: int = 50000, convergence_error: float = 1e-8, log=False):
    n = a.size()[0]
    m = b.size()[0]

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    for it in range(num_iter):
        f_prev = f
        g_prev = g

        # The minus is missing in the paper!
        # It should be equation (7) from https://arxiv.org/pdf/1903.01112.pdf
        rho = -(f.expand_as(C.T).T + g.expand_as(C) - C).clamp(max=0)

        f = (epsilon * a - (rho + g.expand_as(C) - C).sum(1)) / m
        g = (epsilon * b - (rho + f.expand_as(C.T).T - C).sum(0)) / n

        f_diff = (f_prev - f).abs().sum()
        g_diff = (g_prev - g).abs().sum()

        if log:
            print(f"Iteration {it}")
            print(f"f_diff {f_diff}")
            print(f"g_diff {g_diff}")

        if f_diff < convergence_error and g_diff < convergence_error:
            break

    # This is also wrong in the paper, it should be
    #   (f (+) g - C)_+ / epsilon,
    # or, equivalently,
    #   (rho + f (+) g - C) / epsilon.
    # We use the former since it does not require computing rho with
    # the final f and g again.
    return (f.expand_as(C.T).T + g.expand_as(C) - C).clamp(min=0) / epsilon


def sinkhorn_quadratic_gradient_descent(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon: float,
        num_iter: int = 50000, convergence_error: float = 1e-8, log=False):
    n = a.size()[0]
    m = b.size()[0]

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    step = 1.0 / (m + n)

    for it in range(num_iter):
        f_prev = f.clone()
        g_prev = g.clone()

        P = (f.expand_as(C.T).T + g.expand_as(C) - C).clamp(min=0) / epsilon

        f -= step * epsilon * (P.sum(1) - a)
        g -= step * epsilon * (P.sum(0) - b)

        f_diff = (f_prev - f).abs().sum()
        g_diff = (g_prev - g).abs().sum()

        if log:
            print(f"Iteration {it}")
            print(f"f_diff {f_diff}")
            print(f"g_diff {g_diff}")

        if f_diff < convergence_error and g_diff < convergence_error:
            break

    return (f.expand_as(C.T).T + g.expand_as(C) - C).clamp(min=0) / epsilon


def sinkhorn_quadratic_fixed_point_iteration(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon: float,
        num_iter: int = 50000, convergence_error: float = 1e-8, log=False):
    n = a.size()[0]
    m = b.size()[0]

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    for it in range(num_iter):
        f_prev = f.clone()
        g_prev = g.clone()

        P = (f.expand_as(C.T).T + g.expand_as(C) - C).clamp(min=0) / epsilon
        v = - epsilon * (P.sum(1) - a)
        f += (v - v.sum() / (2 * n)) / m
        u = - epsilon * (P.sum(0) - b)
        g += (u - u.sum() / (2 * m)) / n

        f_diff = (f_prev - f).abs().sum()
        g_diff = (g_prev - g).abs().sum()

        if log:
            print(f"Iteration {it}")
            print(f"f_diff {f_diff}")
            print(f"g_diff {g_diff}")

        if f_diff < convergence_error and g_diff < convergence_error:
            break

    return (f.expand_as(C.T).T + g.expand_as(C) - C).clamp(min=0) / epsilon


def sinkhorn_quadratic_nesterov_gradient_descent(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon: float,
        num_iter: int = 50000, convergence_error: float = 1e-8, log=False):
    n = a.size()[0]
    m = b.size()[0]

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    step = 1.0 / (m + n)

    f_previous = f
    g_previous = g

    for it in range(num_iter):
        f_p = f + n * (f - f_previous) / (n + 3)
        g_p = g + n * (g - g_previous) / (n + 3)

        P = (f_p.expand_as(C.T).T
             + g_p.expand_as(C) - C).clamp(min=0) / epsilon

        f_new = f_p - step * epsilon * (P.sum(1) - a)
        g_new = g_p - step * epsilon * (P.sum(0) - b)

        f_diff = (f_previous - f_new).abs().sum()
        g_diff = (g_previous - g_new).abs().sum()

        f_previous = f
        g_previous = g

        f = f_new
        g = g_new

        if log:
            print(f"Iteration {it}")
            print(f"f_diff {f_diff}")
            print(f"g_diff {g_diff}")

        if f_diff < convergence_error and g_diff < convergence_error:
            break

    return (f.expand_as(C.T).T + g.expand_as(C) - C).clamp(min=0) / epsilon



def example_point_clouds(device: torch.device):
    N = [5, 8]
    # N = [200, 300]

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
    C = (X - Y) ** 2

    # epsilon = 9e-4
    epsilon = 1e-2

    start = time.time()

    log = False
    gamma = 1.0 / epsilon
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
