"""
LSFL Aggregator
================
Implements the robust aggregation from:

  "LSFL: A Lightweight and Secure Federated Learning Scheme
   for Edge Computing."  IEEE TIFS, 2023.

Core idea (plaintext equivalent):
  1. Compute the mean update w̄ = (1/N) Σ w_i.
  2. Distance to mean:  d(w_i) = ‖w_i − w̄‖₂  (L2 norm).
  3. Median baseline:   m = median({d(w_1), …, d(w_N)}).
  4. Selection score:   score_i = |d(w_i) − m|.
  5. Keep the top-k clients with the smallest score_i.
  6. Average the selected clients' updates.

Poisoned updates (far from the mean) get large distances; honest clients
cluster near the median distance, so the k nearest to the median are kept.
No cosine similarity, no weighting — hard top-k selection + uniform average.
"""

import numpy as np


def lsfl_aggregate(client_updates, k_frac=0.7):
    """
    LSFL robust aggregation (IEEE TIFS 2023).

    Parameters
    ----------
    client_updates : list of np.ndarray (1-D)
        Gradient deltas from all n clients.
    k_frac : float
        Fraction of clients to keep (top-k nearest to median distance).

    Returns
    -------
    agg : np.ndarray
        Average of the k selected client updates.
    """
    n = len(client_updates)
    G = np.stack(client_updates, axis=0).astype(np.float64)  # (n, d)

    # ---- 1. Mean update ----
    w_mean = np.mean(G, axis=0)

    # ---- 2. Distance to mean (L2 norm) ----
    dists = np.linalg.norm(G - w_mean, axis=1)  # (n,)

    # ---- 3. Median baseline ----
    m = np.median(dists)

    # ---- 4. Selection score: |d(w_i) - m| ----
    scores = np.abs(dists - m)

    # ---- 5. Top-k selection ----
    k = int(round(k_frac * n))
    k = max(1, min(k, n))  # clamp to [1, n]
    selected = np.argsort(scores)[:k]

    # ---- 6. Average selected clients ----
    agg = np.mean(G[selected], axis=0)
    return agg
