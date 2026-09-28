"""
RFed Aggregator
================
Implements the poisoning defense from:

  Miao et al.
  "RFed: Robustness-Enhanced Privacy-Preserving Federated Learning
   Against Poisoning Attack."  IEEE TIFS, 2024.

Core idea (plaintext equivalent):
  The server holds a clean root gradient g0 (trained once on a small
  clean root dataset).  Each client update g_i is scored against g0 with
  the scaled dot-product attention mechanism:

      score_i = (g_i · g0) / sqrt(d)
      weight_i = softmax(score)_i = exp(score_i) / sum_j exp(score_j)
      aggregate = sum_i weight_i · g_i

  Malicious updates point away from g0 → low score → softmax crushes
  their weight.  No client is ever hard-discarded.
"""

import numpy as np


def rfed_aggregate(client_updates, root_gradient):
    """
    RFed robust aggregation (IEEE TIFS 2024).

    Parameters
    ----------
    client_updates : list of np.ndarray (1-D)
        Gradient deltas from all n clients.
    root_gradient : np.ndarray (1-D)
        Root gradient g0 trained on the clean root dataset.

    Returns
    -------
    agg : np.ndarray
        Weighted average of client updates (softmax attention weights).
    """
    n = len(client_updates)
    G = np.stack(client_updates, axis=0).astype(np.float64)  # (n, d)
    d = G.shape[1]
    g0 = root_gradient.astype(np.float64)

    # ---- Scaled dot-product attention scores ----
    scores = np.array([np.dot(g, g0) / np.sqrt(d) for g in G])

    # ---- Softmax (numerically stable) ----
    scores = scores - np.max(scores)       # shift to avoid overflow
    exps = np.exp(scores)
    weights = exps / np.sum(exps)

    # ---- Weighted average ----
    agg = np.sum(weights[:, None] * G, axis=0)
    return agg
