import torch


def sinkhorn_quadratic_cyclic_projection(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon: float,
        num_iter: int = 50000, convergence_error: float = 1e-8,
        log=False) -> torch.Tensor:
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
        num_iter: int = 50000, convergence_error: float = 1e-8,
        log=False) -> torch.Tensor:
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
        num_iter: int = 50000, convergence_error: float = 1e-8,
        log=False) -> torch.Tensor:
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
        num_iter: int = 50000, convergence_error: float = 1e-8,
        log=False) -> torch.Tensor:
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


def sinkhorn_quadratic_nesterov_gradient_descent3(
        cost: torch.Tensor, marg1: torch.Tensor, marg2: torch.Tensor,
        marg3: torch.Tensor, epsilon: float, num_iter: int = 50000,
        convergence_error: float = 1e-8, log=False) -> torch.Tensor:
    n1 = marg1.size()[0]
    n2 = marg2.size()[0]
    n3 = marg3.size()[0]

    p1 = torch.zeros_like(marg1)
    p2 = torch.zeros_like(marg2)
    p3 = torch.zeros_like(marg3)

    step = 1.0 / (n1 + n2 + n3)

    p1_prev = p1
    p2_prev = p2
    p3_prev = p3

    for it in range(num_iter):
        p1_p = p1 + n1 * (p1 - p1_prev) / (n1 + 3)
        p2_p = p2 + n2 * (p2 - p2_prev) / (n2 + 3)
        p3_p = p3 + n3 * (p3 - p3_prev) / (n3 + 3)

        P = (p1.expand_as(cost.T).T
             + p2.expand_as(cost.permute(0, 2, 1)).permute(0, 2, 1)
             + p3.expand_as(cost) - cost).clamp(min=0) / epsilon

        p1_new = p1_p - step * epsilon * (P.sum((1, 2)) - marg1)
        p2_new = p2_p - step * epsilon * (P.sum((0, 2)) - marg2)
        p3_new = p3_p - step * epsilon * (P.sum((0, 1)) - marg3)

        p1_diff = (p1_prev - p1_new).abs().sum()
        p2_diff = (p2_prev - p2_new).abs().sum()
        p3_diff = (p3_prev - p3_new).abs().sum()

        p1_prev = p1
        p2_prev = p2
        p3_prev = p3

        p1 = p1_new
        p2 = p2_new
        p3 = p3_new

        if log:
            print(f"Iteration {it}")
            print(f"p1_diff {p1_diff}")
            print(f"p2_diff {p2_diff}")
            print(f"p3_diff {p3_diff}")

        if p1_diff < convergence_error and p2_diff < convergence_error \
                and p2_diff < convergence_error:
            break

    return (p1.expand_as(cost.T).T
            + p2.expand_as(cost.permute(0, 2, 1)).permute(0, 2, 1)
            + p3.expand_as(cost) - cost).clamp(min=0) / epsilon
