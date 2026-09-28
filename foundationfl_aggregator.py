"""
FoundationFL Aggregation Module
================================
Implements the server-side aggregation algorithm from:
"Do We Really Need to Design New Byzantine-robust Aggregation Rules?"
Minghong Fang et al., NDSS 2025.

Core idea: the server generates synthetic updates by copying the client
update that is "farthest from the extremes," then applies a standard
per-coordinate robust rule (Trimmed-mean or Median) to the augmented pool.

This module is self-contained and requires only numpy. It does NOT modify
any existing code in this project.
"""

import numpy as np


def foundationfl_aggregate(client_updates,
                           m=None,
                           c=None,
                           rule='trimmed_mean'):
    """
    FoundationFL server-side aggregation.

    Parameters
    ----------
    client_updates : list of np.ndarray
        n client gradients, each of shape (d,).  These are the raw
        model-update vectors (Δ_i = θ_local - θ_global) BEFORE any
        compression or noise.
    m : int or None
        Number of synthetic updates to generate.  If None, defaults to
        n // 2 (per the paper's experiments).
    c : int or None
        Number of values to trim from each tail in Trimmed-mean mode.
        If None and rule=='trimmed_mean', defaults to n // 5 (assuming
        ~20 % malicious clients, as in the paper).
    rule : str
        Per-coordinate aggregation rule to apply after augmenting with
        synthetic updates.  Supported: 'trimmed_mean', 'median'.

    Returns
    -------
    np.ndarray
        Aggregated global update, shape (d,).
    """
    n = len(client_updates)
    d = client_updates[0].shape[0]

    # ----- defaults -----
    if m is None:
        m = max(1, n // 2)          # synthetic copies: half of client count

    G = np.stack(client_updates, axis=0)   # (n, d)

    # ----------------------------------------------------------------
    # Step 1 – per-dimension extreme vectors
    # ----------------------------------------------------------------
    g_max = np.max(G, axis=0)       # shape (d,)
    g_min = np.min(G, axis=0)       # shape (d,)

    # ----------------------------------------------------------------
    # Step 2 – closeness-to-extremes score for each client
    # ----------------------------------------------------------------
    scores = np.empty(n)
    for i in range(n):
        dist_max = float(np.linalg.norm(G[i] - g_max))
        dist_min = float(np.linalg.norm(G[i] - g_min))
        scores[i] = min(dist_max, dist_min)

    # ----------------------------------------------------------------
    # Step 3 – select the client farthest from both extremes
    # ----------------------------------------------------------------
    i_star = int(np.argmax(scores))

    # ----------------------------------------------------------------
    # Step 4 – generate m synthetic copies of the safe update
    # ----------------------------------------------------------------
    synthetic = np.tile(G[i_star], (m, 1))          # (m, d)

    # ----------------------------------------------------------------
    # Step 5 – pool = real + synthetic
    # ----------------------------------------------------------------
    all_updates = np.concatenate([G, synthetic], axis=0)   # (n+m, d)

    # ----------------------------------------------------------------
    # Step 6 – per-coordinate robust aggregation
    # ----------------------------------------------------------------
    aggregated = np.empty(d, dtype=G.dtype)

    if rule == 'trimmed_mean':
        if c is None:
            c = max(1, n // 5)       # ~20 % malicious
        lo = c
        hi = (n + m) - c
        for k in range(d):
            vals = np.sort(all_updates[:, k])
            aggregated[k] = np.mean(vals[lo:hi])

    elif rule == 'median':
        for k in range(d):
            aggregated[k] = np.median(all_updates[:, k])

    else:
        raise ValueError(f"Unknown aggregation rule: {rule}")

    return aggregated
