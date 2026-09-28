"""
FoolsGold / Sybil-Resilient Aggregator
========================================
Fung, Yoon, Beschastnikh. "Mitigating Sybils in Federated Learning
Poisoning."  RAID 2020.

Core idea: Sybil attackers submit highly similar gradient updates, while
honest clients produce diverse updates due to unique local data.

Algorithm (per round):
  1. Accumulate historical gradients:  H_i += g_i
  2. Pairwise cosine similarity:  cs[i][j] = cos_sim(H_i, H_j)
  3. Max similarity per client:  v[i] = max_j cs[i][j]
  4. Pardoning:  if v[j] > v[i], cs[i][j] *= v[i] / v[j]
  5. Sybil score → confidence:  alpha[i] = 1 - max_j cs[i][j]
  6. Normalise:  alpha /= max(alpha)
  7. Logit:  alpha[i] = ln(alpha[i] / (1-alpha[i])) + 0.5, clip [0,1]
  8. Aggregate:  sum_i alpha[i] * g_i
"""

import numpy as np


def _cosine_sim(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def sybil_aggregate(client_updates, history, eps=1e-8):
    """
    FoolsGold robust aggregation (RAID 2020).

    Parameters
    ----------
    client_updates : list of np.ndarray (1-D)
        Gradient vectors from all n clients for this round.
    history : list of np.ndarray or None
        Cumulative historical gradients, shape (n, d).  Updated in-place.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    agg : np.ndarray        Weighted sum of client updates.
    history : list          Updated cumulative history (same list object).
    """
    n = len(client_updates)
    d = len(client_updates[0])
    G = np.stack(client_updates, axis=0).astype(np.float64)  # (n, d)

    # ---- 1. Accumulate historical gradients ----
    if history is None or len(history) != n:
        history = [np.zeros(d, dtype=np.float64) for _ in range(n)]
    for i in range(n):
        history[i] = history[i] + G[i]

    # ---- 2. Pairwise cosine similarity (using accumulated history) ----
    cs = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_sim(history[i], history[j])
            cs[i][j] = sim
            cs[j][i] = sim

    # ---- 3. Max similarity per client ----
    v = np.max(cs, axis=1)  # (n,)

    # ---- 4. Pardoning ----
    cs_adapt = cs.copy()
    for i in range(n):
        for j in range(n):
            if v[j] > v[i]:
                cs_adapt[i][j] = cs[i][j] * (v[i] / max(v[j], eps))

    # ---- 5. Confidence:  honest → low similarity → high alpha ----
    alpha = 1.0 - np.max(cs_adapt, axis=1)  # (n,)

    # ---- 6. Normalise ----
    max_alpha = np.max(alpha)
    if max_alpha > eps:
        alpha = alpha / max_alpha
    # else all-zero → fallback to equal weights

    # ---- 7. Logit (amplify divergence, Appendix A) ----
    # logit(x) = ln(x/(1-x)) + 0.5, centred at 0.5
    for i in range(n):
        a = alpha[i]
        if a <= 0.0:
            alpha[i] = 0.0
        elif a >= 1.0:
            alpha[i] = 1.0
        else:
            val = np.log(a / (1.0 - a)) + 0.5
            alpha[i] = float(np.clip(val, 0.0, 1.0))

    # ---- 8. Weighted sum (not average — paper uses sum of alpha*g) ----
    total_alpha = np.sum(alpha)
    if total_alpha < eps:
        alpha = np.ones(n) / n
        total_alpha = 1.0

    agg = np.sum(alpha[:, None] * G, axis=0) / total_alpha
    return agg, history
