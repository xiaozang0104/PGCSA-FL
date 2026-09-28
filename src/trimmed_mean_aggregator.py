"""
Trimmed Mean / Median Aggregation Module
==========================================
Implements the per-coordinate Byzantine-robust aggregation rules from:
"Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
Fang, Cao, Jia, Gong, USENIX Security 2020.

These rules are originally from:
"Byzantine-Tolerant Machine Learning"
Yin, Chen, Kannan, Bartlett, 2018.

Rules:
  1. Trimmed Mean: For each coordinate j, sort the values across all
     clients, trim the largest f and smallest f values, and average
     the remaining (n - 2f) values.
  2. Median: For each coordinate j, take the median of all client values.

This module is self-contained and requires only numpy.
"""

import numpy as np


def trimmed_mean_aggregate(client_updates, f):
    """
    Per-coordinate trimmed mean aggregation.

    For each parameter dimension, sort the n client values across that
    dimension, remove the f largest and f smallest, then average the
    remaining n-2f values.

    Parameters
    ----------
    client_updates : list of np.ndarray, each shape (d,)
        n client model deltas.
    f : int
        Number of presumed Byzantine clients to trim from each tail.

    Returns
    -------
    np.ndarray
        Aggregated global update, shape (d,).
    """
    n = len(client_updates)
    G = np.stack(client_updates, axis=0)  # (n, d)

    if f == 0:
        return np.mean(G, axis=0)

    # If trimming would leave nothing, fall back to plain mean
    if n - 2 * f <= 0:
        return np.mean(G, axis=0)

    sorted_G = np.sort(G, axis=0)
    trimmed = sorted_G[f:n - f, :]
    return np.mean(trimmed, axis=0)


def median_aggregate(client_updates):
    """
    Per-coordinate median aggregation.

    For each parameter dimension, compute the median of all client values
    in that dimension.

    Parameters
    ----------
    client_updates : list of np.ndarray, each shape (d,)
        n client model deltas.

    Returns
    -------
    np.ndarray
        Aggregated global update, shape (d,).
    """
    G = np.stack(client_updates, axis=0)  # (n, d)
    return np.median(G, axis=0)
