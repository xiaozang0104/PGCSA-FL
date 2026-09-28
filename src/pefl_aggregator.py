"""
PEFL Aggregator
================
Implements the poisoning defense from:

  Liu, Ma, Aafer, Lee, Zhai, Wang, Zhang.
  "Privacy-Enhanced Federated Learning Against Poisoning Adversaries."
  IEEE TIFS, 2021.

Core idea:
  1. Compute coordinate-wise median of all client gradients → benchmark
  2. Pearson correlation between each client's gradient and the benchmark
  3. Fisher z-transform to map correlation → weight: μ = max{0, ln((1+ρ)/(1-ρ)) - 0.5}
  4. Weighted average: Σ μ_x·G_x / Σ μ_x

A client whose gradient points in a very different direction from the median
gets ρ close to 0 or negative, μ → 0, and is excluded from aggregation.
"""

import numpy as np


def _pearson_corr(a, b):
    """Pearson correlation coefficient between two 1-D vectors."""
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    cov = np.dot(a_centered, b_centered)
    norm_a = np.linalg.norm(a_centered)
    norm_b = np.linalg.norm(b_centered)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return cov / (norm_a * norm_b)


def pefl_aggregate(client_updates):
    """
    PEFL robust aggregation (IEEE TIFS 2021).

    Parameters
    ----------
    client_updates : list of np.ndarray (1-D)
        Gradient deltas from all n clients.

    Returns
    -------
    agg : np.ndarray
        Weighted average of client updates.
    """
    n = len(client_updates)
    G = np.stack(client_updates, axis=0).astype(np.float64)  # (n, d)

    # ---- 1. Coordinate-wise median (benchmark) ----
    median = np.median(G, axis=0)  # (d,)

    # ---- 2. Pearson correlation per client ----
    weights = np.zeros(n)
    for i in range(n):
        rho = _pearson_corr(G[i], median)
        # ---- 3. Fisher z-transform with threshold 0.5 ----
        if rho >= 1.0:
            rho = 1.0 - 1e-12
        if rho <= -1.0:
            rho = -1.0 + 1e-12
        ratio = (1.0 + rho) / (1.0 - rho)
        if ratio <= 0:
            mu = 0.0
        else:
            mu = max(0.0, np.log(ratio) - 0.5)
        weights[i] = mu

    total_w = np.sum(weights)

    # ---- 4. Weighted average ----
    if total_w < 1e-12:
        # All clients excluded → fall back to median
        return median

    agg = np.sum(weights[:, None] * G, axis=0) / total_w
    return agg
