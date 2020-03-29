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
