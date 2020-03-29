from typing import Union

import torch
from .regular import sinkhorn


def sinkhorn_tsallis_entropy(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, q: float,
        epsilon: float, num_iter: int = 1000,
        convergence_error: float = 1e-2, log=False) -> torch.Tensor:

    assert q >= 0

    if q == 1:
        return sinkhorn(C, a, b, epsilon, num_iter, convergence_error)

    elif q < 1:
        return _sinkhorn_tsallis_entropy_sor(
            C, a, b, q, epsilon, num_iter, convergence_error, log)
    else:
        return _sinkhorn_tsallis_kl_proj_descent(
            C, a, b, q, epsilon, num_iter, convergence_error, log)


def _sinkhorn_tsallis_entropy_sor(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, q: float,
        epsilon: float, num_iter: int = 1000,
        convergence_error: float = 1e-2, log=False) -> torch.Tensor:

    def q_exp(x: Union[torch.Tensor, float]):
        return (1 + (1 - q) * x) ** (1 / (1 - q))

    n = a.size()[0]
    m = b.size()[0]

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    A = epsilon * C
    A = A - A.min(dim=0).values
    A = (A.T - A.min(dim=1).values).T

    q1 = q_exp(-1)

    for it in range(num_iter):
        P = q1 / q_exp(A)

        if it > 0:
            a_diff = (P.sum(1) / a - 1).abs().max()
            b_diff = (P.sum(0) / b - 1).abs().max()

            if log:
                print(f"Iteration {it}")
                print(f"a_diff {a_diff}")
                print(f"b_diff {b_diff}")

            if a_diff < convergence_error and b_diff < convergence_error:
                break

        Ap = 1 + (1 - q) * A
        P_1 = P / Ap
        P_2 = P_1 / Ap

        d = P.sum(dim=1) - a
        u = (1 - q / 2.0) * P_2.sum(dim=1)
        v = -P_1.sum(dim=1)

        delta = v ** 2 - 4 * u * d

        for i in range(n):
            if delta[i] >= 0 and d[i] < 0 < u[i]:
                f[i] = - (v[i] + torch.sqrt(delta[i])) / (2 * u[i])
            elif u[i] != 0:
                f[i] = -2 * d[i] / v[i]
            else:
                f[i] = 0

            max_value = 1 / (2 * (1 - q) * Ap[i, :].max())
            if f[i].abs() > max_value:
                f[i] = d[i].sign() * max_value

        A += f.expand_as(A.T).T

        P = q1 / q_exp(A)
        Ap = 1 + (1 - q) * A
        P_1 = P / Ap
        P_2 = P_1 / Ap

        d = P.sum(dim=0) - b
        u = (1 - q / 2.0) * P_2.sum(dim=0)
        v = -P_1.sum(dim=0)

        delta = v ** 2 - 4 * u * d

        for j in range(m):
            if delta[j] >= 0 and d[j] < 0 < v[j]:
                g[j] = - (v[j] + torch.sqrt(delta[j])) / (2 * u[j])
            elif u[j] != 0:
                g[j] = -2 * d[j] / v[j]
            else:
                g[j] = 0

            max_value = 1 / (2 * (1 - q) * Ap[:, j].max())
            if g[j].abs() > max_value:
                g[j] = d[j].sign() * max_value

        A += g.expand_as(A)

    P = q1 / q_exp(A)

    return P


def _sinkhorn_tsallis_kl_proj_descent(
        C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, q: float,
        epsilon: float, num_iter: int = 1000,
        rate=1, log=False) -> torch.Tensor:

    # This one does not work

    assert q > 1

    P = a.reshape(-1, 1) * b.reshape(1, -1)

    def objective():
        P_q = P**q
        return (P_q * C).norm() - (P_q - P).sum() / ((1 - q) * epsilon)

    best_score = objective()
    best_P = P

    for it in range(num_iter):
        P_grad = C + (q * P**(q - 1) - 1) / ((1 - q) * epsilon)
        T = (-rate / (it + 1) * P_grad).exp()
        P = sinkhorn(P * T, a, b, 1.0 / epsilon, C_is_gibbs_kernel=True)

        new_score = objective()
        if log:
            print(f"Iteration {it}")
            print(f"Score: {new_score}")

        if new_score < best_score:
            best_score = new_score
            best_P = P

    return best_P
