import torch


def sinkhorn(C: torch.Tensor, a: torch.Tensor, b: torch.tensor,
             epsilon: float, num_iter: int = 1000,
             convergence_error: float = 1e-8,
             log=False, C_is_gibbs_kernel=False) -> torch.Tensor:

    if C_is_gibbs_kernel:
        K = C
    else:
        K = torch.exp(-C / epsilon)

    v = torch.ones_like(b)
    u = torch.ones_like(a)

    # def transform_C_f_g(f, g):
    #     return (C - f.expand_as(C.T).T - g.expand_as(C)) / epsilon
    #
    # def calculate_loss(f, g):
    #     return -(f @ a + g @ b
    #              - epsilon * torch.exp(
    #                 transform_C_f_g(f, g)).sum())

    for i in range(num_iter):
        u_prev = u
        v_prev = v
        u = a / (K @ v)
        v = b / (K.T @ u)

        u_diff = (u_prev - u).abs().sum()
        v_diff = (v_prev - v).abs().sum()

        if log:
            print(f"Iteration {i}")
            print(f"u_diff={u_diff}")
            print(f"v_diff={v_diff}")

        if u_diff < convergence_error and v_diff < convergence_error:
            break

    return torch.diag(u) @ K @ torch.diag(v)


def sinkhorn3(cost: torch.Tensor, marg1: torch.Tensor, marg2: torch.Tensor,
              marg3: torch.Tensor, epsilon: float, num_iter: int = 500,
              convergence_error: float = 1e-8, log=False) -> torch.Tensor:

    K = torch.exp(-cost / epsilon)

    u = torch.ones_like(marg1)
    v = torch.ones_like(marg2)
    w = torch.ones_like(marg3)

    for i in range(num_iter):
        u_prev = u
        v_prev = v
        w_prev = w

        u = marg1 / (K @ w @ v)
        v = marg2 / (K.permute(1, 2, 0) @ u @ w)
        w = marg3 / (K.permute(2, 1, 0) @ u @ v)

        u_diff = (u_prev - u).abs().sum()
        v_diff = (v_prev - v).abs().sum()
        w_diff = (w_prev - w).abs().sum()

        if log:
            print(f"Iteration {i}")
            print(f"u_diff={u_diff}")
            print(f"v_diff={v_diff}")
            print(f"w_diff={w_diff}")

        if u_diff < convergence_error and v_diff < convergence_error \
                and w_diff < convergence_error:
            break

    return K.matmul(torch.diag(w))\
            .permute(0, 2, 1)\
            .matmul(torch.diag(v))\
            .permute(1, 2, 0)\
            .matmul(torch.diag(u)).T
