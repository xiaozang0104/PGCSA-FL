"""
Split Aggregation Aggregator
=============================
Implements the robust aggregation from:

  Lu et al.
  "Split Aggregation: Lightweight Privacy-Preserving Federated Learning
   Resistant to Byzantine Attacks."  IEEE TIFS, 2024.

Core idea (plaintext equivalent):
  1. Center gradients:  Gc = g_i - mean(g).
  2. SVD top right singular vector v (dominant variation direction).
  3. Outlier score:  tau_i = <Gc_i, v>^2  (squared projection).
  4. Tolerance:  eps = tau_med - tau_min  (median minus min).
     Threshold:  tau_med + eps = 2*tau_med - tau_min.
  5. Weight:  coef_i = tanh(1 - tau_i/sum(tau)) if tau_i <= threshold,
             else 0.
     Normalize:  alpha_i = coef_i / sum(coef).
  6. Aggregate:  weighted average of surviving clients.

Poisoned updates project strongly onto v (large tau_i) and are
down-weighted or zeroed.  Honest clients cluster at small tau_i.
"""

import numpy as np


def split_aggregate(client_updates):
    """
    Split Aggregation robust aggregation (IEEE TIFS 2024).

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

    # ---- 1. Center gradients ----
    Gc = G - np.mean(G, axis=0)  # (n, d)

    # ---- 2. Top right singular vector ----
    # Gc is (n, d) with small n; full SVD is cheap (O(n^2 d)).
    _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    v = Vt[0, :]  # (d,) dominant direction

    # ---- 3. Outlier scores (squared projection onto v) ----
    tau = (Gc @ v) ** 2  # (n,)

    # ---- 4. Tolerance / threshold ----
    tau_med = np.median(tau)
    tau_min = np.min(tau)
    eps = tau_med - tau_min
    threshold = tau_med + eps  # = 2*tau_med - tau_min

    # ---- 5. Adaptive weights (tanh soft weighting + hard threshold) ----
    sum_tau = np.sum(tau)
    coefs = np.zeros(n)
    for i in range(n):
        if tau[i] <= threshold:
            if sum_tau > 1e-12:
                x = 1.0 - tau[i] / sum_tau
            else:
                x = 1.0
            coefs[i] = np.tanh(x)

    total_coef = np.sum(coefs)
    if total_coef < 1e-12:
        alpha = np.ones(n) / n
    else:
        alpha = coefs / total_coef

    # ---- 6. Weighted average ----
    agg = np.sum(alpha[:, None] * G, axis=0)
    return agg
