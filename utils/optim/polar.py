"""Approximate polar factorization (orthogonalization).

newton_schulz maps a matrix G to an approximation of its polar factor UV^T,
where G = U S V^T, using Keller Jordan's quintic Newton-Schulz iteration (Muon).
"""
import torch


@torch.compile
def newton_schulz(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.mT
    return X
