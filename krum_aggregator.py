"""
Krum Aggregation Module
========================
Implements the Krum / Multi-Krum Byzantine-robust aggregation rule from:
"Machine Learning with Adversaries: Byzantine-Tolerant Gradient Descent"
Blanchard, El Mhamdi, Guerraoui, Stainer, NIPS 2017.

Core idea:
  1. For each client i, compute squared Euclidean distance to every other
     client j, select the n-f-2 closest (smallest) distances, and sum them
     as the score s(i).
  2. Krum: output the single gradient with the smallest score.
  3. Multi-Krum (m = n-f): average the m gradients with the m smallest
     scores.  This is the default (converges faster than pure Krum).

Constraint: 2f + 2 < n  (n > 2f + 2)
"""

import numpy as np


def krum_aggregate(client_updates, f, multi=True):
    """
    Krum / Multi-Krum Byzantine-robust aggregation.

    Parameters
    ----------
    client_updates : list of np.ndarray, each shape (d,)
        n client model deltas.
    f : int
        Number of presumed Byzantine clients.
    multi : bool
        If True (default), use Multi-Krum with m = n-f.
        If False, use pure Krum (return single best gradient).

    Returns
    -------
    np.ndarray
        Aggregated global update, shape (d,).
    """
    n = len(client_updates)
    d = client_updates[0].shape[0]

    # Constraint check
    if n <= 2 * f + 2:
        raise ValueError(
            f"Krum requires n > 2f+2 (n={n}, f={f}, 2f+2={2*f+2})")

    G = np.stack(client_updates, axis=0)  # (n, d)

    # --- compute all pairwise squared Euclidean distances ---
    # ||G[i] - G[j]||^2 = ||G[i]||^2 + ||G[j]||^2 - 2 * G[i]·G[j]
    norms = np.sum(G ** 2, axis=1)                # (n,)
    dots = G @ G.T                                # (n, n)
    dist_sq = (norms[:, None] + norms[None, :]
               - 2 * dots)                        # (n, n)
    np.fill_diagonal(dist_sq, np.inf)             # ignore self

    k = n - f - 2  # number of closest neighbors to keep

    # --- Krum scores ---
    # For each i: sort distances, sum the k smallest
    sorted_idx = np.argsort(dist_sq, axis=1)       # (n, n)
    scores = np.empty(n)
    for i in range(n):
        closest = sorted_idx[i, :k]
        scores[i] = dist_sq[i, closest].sum()

    # --- select best gradient(s) ---
    if multi:
        # Multi-Krum: average the m = n-f gradients with smallest scores
        m = n - f
        best_idx = np.argsort(scores)[:m]          # m smallest scores
        return np.mean(G[best_idx], axis=0)
    else:
        # Pure Krum: single best gradient
        best_idx = int(np.argmin(scores))
        return G[best_idx].copy()
