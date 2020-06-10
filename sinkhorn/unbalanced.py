import torch


def sinkhorn_unbalanced(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
        epsilon: float, tau: float, num_iter: int = 5000,
        convergence_error: float = 1e-8, log=False) -> torch.Tensor:

    assert a.shape == b.shape

    n = a.shape[0]

    n_t = torch.tensor(n, device=C.device, dtype=torch.float)

    log_n = torch.log(n_t)

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    alpha = a.sum()
    beta = b.sum()

    S = (alpha + beta) / 2 + 0.5 + 1 / (4 * log_n)
    T = (alpha + beta) / 2 * \
        (torch.log((alpha + beta) / 2) + 2 * log_n - 1) + log_n + 2.5

    U = max(S + T,
            torch.tensor(2 * epsilon, device=C.device),
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


def sinkhorn_unbalanced3(cost: torch.Tensor, marg1: torch.Tensor,
                         marg2: torch.Tensor, marg3: torch.Tensor,
                         epsilon: float, tau: float, num_iter: int = 500,
                         convergence_error: float = 1e-8,
                         log: bool = False) -> torch.Tensor:
    assert marg1.shape == marg2.shape == marg3.shape

    n = marg1.shape[0]

    n_t = torch.tensor(n, device=cost.device, dtype=torch.float)

    log_n = torch.log(n_t)

    p1 = torch.zeros_like(marg1)
    p2 = torch.zeros_like(marg2)
    p3 = torch.zeros_like(marg3)

    marg1_sum = marg1.sum()
    marg2_sum = marg2.sum()
    marg3_sum = marg3.sum()

    marg_sums_mean = (marg1_sum + marg2_sum + marg3_sum) / 3

    # These three constants were just "guessed" by basically changing
    # the occurrences of "2" with "3" in the formulas
    S = marg_sums_mean + 1 / 3 + 1 / (9 * log_n)
    T = marg_sums_mean * (marg_sums_mean.log() + 3 * log_n - 1) + log_n + 2.5

    U = max(S + T,
            torch.tensor(2 * epsilon, device=cost.device),
            (9 * epsilon * log_n) / tau,
            (9 * epsilon * (marg1_sum + marg2_sum + marg3_sum) * log_n) / tau)

    eta = epsilon / U

    A = cost / eta
    A = A - A.min(dim=0).values
    A = (A.permute(1, 0, 2) - A.min(dim=1).values).permute(1, 0, 2)
    A = (A.permute(2, 0, 1) - A.min(dim=2).values).permute(1, 2, 0)

    K = torch.exp(-A)

    scale_factor = (eta * tau) / (eta + tau)

    def compute_coupling_matrix():
        return K.matmul(torch.diag((p3 / eta).exp())) \
            .permute(0, 2, 1) \
            .matmul(torch.diag((p2 / eta).exp())) \
            .permute(1, 2, 0) \
            .matmul(torch.diag((p1 / eta).exp())).T

    P = compute_coupling_matrix()

    marg1_log = marg1.log()
    marg2_log = marg2.log()
    marg3_log = marg3.log()

    for it in range(num_iter):
        p1_prev = p1
        p2_prev = p2
        p3_prev = p3

        p1_k = P.sum((1, 2)) + 1e-9
        p1 = (p1 / eta + marg1_log - p1_k.log()) * scale_factor
        P = compute_coupling_matrix()

        p2_k = P.sum((0, 2)) + 1e-9
        p2 = (p2 / eta + marg2_log - p2_k.log()) * scale_factor
        P = compute_coupling_matrix()

        p3_k = P.sum((0, 1)) + 1e-9
        p3 = (p3 / eta + marg3_log - p3_k.log()) * scale_factor
        P = compute_coupling_matrix()

        p1_diff = (p1_prev - p1).abs().sum()
        p2_diff = (p2_prev - p2).abs().sum()
        p3_diff = (p3_prev - p3).abs().sum()

        if log:
            print(f"Iteration {it}")
            print(f"p1_diff {p1_diff}")
            print(f"p2_diff {p2_diff}")
            print(f"p3_diff {p3_diff}")

        if p1_diff < convergence_error and p2_diff < convergence_error \
                and p3_diff < convergence_error:
            break

    return P

