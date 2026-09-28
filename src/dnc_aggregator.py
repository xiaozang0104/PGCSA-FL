"""
DnC (Divide-and-Conquer) Aggregator
=====================================
Implements the robust aggregation defense from:

  Shejwalkar, Houmansadr, Kairouz, Ramage.
  "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and
   Defenses for Federated Learning."  NDSS 2021.

Algorithm (per round):
  1. Randomly subsample b gradient dimensions
  2. Form n x b centered matrix G̃_c (subtract dimension-wise mean)
  3. Compute top right singular vector v (first principal component)
  4. Score: s_i = <g̃_i - μ̃, v>^2
  5. Keep the (n - c*m) gradients with lowest scores → surviving set I
  6. Repeat for n_iters rounds; intersection of all I → I_final
  7. Average full-dimensional original gradients indexed by I_final
"""

import numpy as np


def dnc_aggregate(client_updates, m, n_iters=1, c=1, b=10000):
    """
    DnC robust aggregation (NDSS 2021).

    Parameters
    ----------
    client_updates : list of np.ndarray (1-D)
        Gradient vectors from all n clients, each of length d.
    m : int
        Upper bound on the number of malicious clients.
    n_iters : int
        Number of random subsampling rounds (default 1 for iid).
    c : float
        Filtering fraction; c*m gradients are removed each round.
    b : int
        Dimensionality of each random subsample.

    Returns
    -------
    agg : np.ndarray
        Aggregated gradient (average of survivors, full-dimensional).
    """
    n = len(client_updates)
    d = len(client_updates[0])
    G_full = np.stack(client_updates, axis=0)  # (n, d)

    # 无恶意客户端（m=0，如恶意比例 0%）：直接 FedAvg，跳过 DnC 机制
    if m <= 0:
        return np.mean(G_full, axis=0)

    # ---- Clamp b to actual gradient dimension ----
    b = min(b, d)

    survivors = set(range(n))

    for _ in range(n_iters):
        # (a) Random subsample b dimensions
        if b < d:
            r = np.sort(np.random.choice(d, size=b, replace=False))
        else:
            r = np.arange(d)

        G_sub = G_full[:, r]  # (n, b)

        # (b) Center: subtract dimension-wise mean
        mu = G_sub.mean(axis=0, keepdims=True)  # (1, b)
        G_c = G_sub - mu                         # (n, b)

        # (c) Top right singular vector
        #     n=20, b≤10000 → np.linalg.svd is O(n²·b) = cheap
        _, _, Vt = np.linalg.svd(G_c, full_matrices=False)
        v = Vt[0, :]  # first right singular vector (b,)

        # (d) Scores: s_i = <g̃_i - μ̃, v>²
        scores = (G_c @ v) ** 2  # (n,)

        # (e) Keep (n - c*m) lowest scores
        #     kth 必须 < n，否则 np.argpartition 越界崩溃
        k_keep = max(1, min(n - 1, n - int(np.ceil(c * m))))
        keep_idx = np.argpartition(scores, k_keep)[:k_keep]
        survivors &= set(keep_idx)

        if len(survivors) == 0:
            survivors = set(range(n))
            break

    # (g) Average full-dimensional gradients of survivors
    survivors_list = sorted(survivors)
    agg = np.mean(G_full[survivors_list], axis=0)
    return agg
