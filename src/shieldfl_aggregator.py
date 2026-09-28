"""
ShieldFL Aggregator
====================
Implements the poisoning defense from:

  Ma, Ma, Miao, Li, Deng.
  "ShieldFL: Mitigating Model Poisoning Attacks in
   Privacy-Preserving Federated Learning."  IEEE TIFS, 2022.

Core idea (plaintext equivalent of the privacy-preserving protocol):
  1. Round 1: plain average (assumes clean first round).
  2. Round t > 1:
     a. Compute cosine similarity of each gradient to the previous
        round's aggregate g(t-1).
     b. Pick the "poisonous baseline" g* = the gradient most dissimilar
        to g(t-1).
     c. Confidence: eta_i = 1 - cos(g_i, g*).  A client similar to the
        baseline (likely malicious) gets low weight; a client dissimilar
        to it (likely benign) gets high weight.
     d. Weighted average: g(t) = sum(eta_i/sum(eta) * g_i).

No hard threshold is used — soft reweighting avoids misclassifying
non-IID benign clients.
"""

import numpy as np


def _cosine(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def shieldfl_aggregate(client_updates, prev_agg):
    """
    ShieldFL robust aggregation (IEEE TIFS 2022).

    Parameters
    ----------
    client_updates : list of np.ndarray (1-D)
        Gradient deltas from all n clients.
    prev_agg : np.ndarray or None
        Aggregated gradient from the previous round.  None → round 1.

    Returns
    -------
    agg : np.ndarray       Aggregated gradient (plain average on round 1).
    new_prev_agg : np.ndarray  Same as agg (stored for next round).
    """
    n = len(client_updates)
    G = np.stack(client_updates, axis=0).astype(np.float64)  # (n, d)

    # ---- Round 1: plain average ----
    if prev_agg is None:
        agg = np.mean(G, axis=0)
        return agg, agg

    # ---- Round t > 1 ----
    # (a) cosine to previous aggregate
    cos_prev = np.array([_cosine(g, prev_agg) for g in G])

    # (b) poisonous baseline = most dissimilar to previous aggregate
    g_star = G[int(np.argmin(cos_prev))]

    # (c) confidence: similarity to baseline → low weight
    cos_base = np.array([_cosine(g, g_star) for g in G])
    eta = 1.0 - cos_base            # in [0, 2]
    eta = np.clip(eta, 0.0, None)

    total = np.sum(eta)
    if total < 1e-12:
        weights = np.ones(n) / n
    else:
        weights = eta / total

    # (d) weighted average
    agg = np.sum(weights[:, None] * G, axis=0)
    return agg, agg
