"""
FLTrust Aggregator
===================
Implements the Byzantine-robust aggregation from:

  Cao, Fang, Liu, Gong.
  "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping."
  NDSS 2021.

Core idea: The server maintains a small clean "root dataset" and computes
its own reference gradient g_0 on it.  Client gradients are compared to
g_0 via cosine similarity; those pointing in similar directions are
trusted; those pointing opposite are dropped (ReLU).

Algorithm (per round):
  1. Server trains on root data → g_0
  2. Client i trains on local data → g_i
  3. Cosine similarity:  c_i = <g_0, g_i> / (||g_0||·||g_i||)
  4. Trust score:  TS_i = max(0, c_i)
  5. Normalise magnitude:  g_i' = g_i · ||g_0|| / ||g_i||
  6. Aggregate:  g = Σ TS_i · g_i' / Σ TS_i
"""

import numpy as np


def fltrust_aggregate(client_updates, server_update):
    """
    FLTrust robust aggregation (NDSS 2021).

    Parameters
    ----------
    client_updates : list of np.ndarray (1-D)
        Gradient deltas from all n clients.
    server_update : np.ndarray (1-D)
        Server's own model update trained on the root dataset (g_0).

    Returns
    -------
    agg : np.ndarray
        Weighted average of client updates, aligned with server direction.
    """
    n = len(client_updates)
    G = np.stack(client_updates, axis=0)  # (n, d)
    g0 = server_update.astype(np.float64)

    norm_g0 = np.linalg.norm(g0)
    if norm_g0 < 1e-12:
        # server update is zero → fall back to plain average
        return np.mean(G, axis=0)

    # ---- Cosine similarity & Trust Score ----
    ts = np.zeros(n)
    norms_g = np.linalg.norm(G, axis=1)  # (n,)

    for i in range(n):
        if norms_g[i] < 1e-12:
            ts[i] = 0.0
            continue
        cos_i = np.dot(g0, G[i]) / (norm_g0 * norms_g[i])
        ts[i] = max(0.0, cos_i)  # ReLU

    total_ts = np.sum(ts)

    if total_ts < 1e-12:
        # All clients have TS=0 → use server update alone
        return g0.astype(G.dtype)

    # ---- Magnitude normalisation: g_i' = g_i · ||g_0|| / ||g_i|| ----
    G_norm = G.copy().astype(np.float64)
    for i in range(n):
        if norms_g[i] > 1e-12:
            G_norm[i] = G[i] * (norm_g0 / norms_g[i])

    # ---- Weighted average ----
    agg = np.sum(ts[:, None] * G_norm, axis=0) / total_ts
    return agg.astype(G.dtype)
