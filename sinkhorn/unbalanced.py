import torch


def sinkhorn_unbalanced(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
        epsilon: float, tau: float, num_iter: int = 5000,
        convergence_error: float = 1e-8, log=False) -> torch.Tensor:

    assert a.shape == b.shape

    n = a.shape[0]

    n_t = torch.Tensor([n], device=C.device)

    log_n = torch.log(n_t)

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    alpha = a.sum()
    beta = b.sum()

    S = (alpha + beta) / 2 + 0.5 + 1 / (4 * log_n)
    T = (alpha + beta) / 2 * \
        (torch.log((alpha + beta) / 2) + 2 * log_n - 1) + log_n + 2.5

    U = max(S + T,
            torch.Tensor([2 * epsilon], device=C.device),
            (4 * epsilon * log_n) / tau,
            (4 * epsilon * (alpha + beta) * log_n) / tau)

    eta = epsilon / U

    A = C / eta
    A = A - A.min(dim=0).values
    A = (A.T - A.min(dim=1).values).T

    K = torch.exp(-A)

    scale_factor = (eta * tau) / (eta + tau)

    P = torch.diag((f / eta).exp()) @ K @ torch.diag((g / eta).exp())

    for it in range(num_iter):
        f_prev = f
        g_prev = g

        a_k = P.sum(1)
        f = (f / eta + a.log() - a_k.log()) * scale_factor
        P = torch.diag((f / eta).exp()) @ K @ torch.diag((g / eta).exp())

        b_k = P.sum(0)
        g = (g / eta + b.log() - b_k.log()) * scale_factor
        P = torch.diag((f / eta).exp()) @ K @ torch.diag((g / eta).exp())

        f_diff = (f_prev - f).abs().sum()
        g_diff = (g_prev - g).abs().sum()

        if log:
            print(f"Iteration {it}")
            print(f"f_diff {f_diff}")
            print(f"g_diff {g_diff}")

        if f_diff < convergence_error and g_diff < convergence_error:
            break

    return P
