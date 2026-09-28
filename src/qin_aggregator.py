"""
Qin et al. (TIFS 2026) Aggregation Module
==========================================
Implements the FLTrust-style Byzantine-robust defense from:
"Efficient Byzantine-Robust Privacy-Preserving Federated Learning
 via Dimension Compression"
Xian Qin, Xue Yang, Xiaohu Tang, IEEE TIFS 2026.

Core idea:
  1. Server holds a small trusted dataset D_s (clean, balanced).
  2. Each round, compute a "reference gradient" g_std on D_s.
  3. (Optionally) JL-compress both client gradients and g_std.
  4. For each client, compute cosine similarity between g_i and g_std.
  5. Trust score TS_i = max(0, cos_i) — negative cosine = zero trust.
  6. Weighted aggregation: alpha_i = (TS_i * ||g_std|| / max(||g_i||, eps))
                               / sum_j TS_j
  7. Global update = sum_i alpha_i * g_i

NOTE: This module omits the Paillier HE / dual-server masking from the
original paper, because cryptographic privacy has no effect on accuracy.
The defense logic (JL compression + FLTrust cosine similarity +
weighted aggregation) is preserved in full.
"""

import numpy as np
import torch
from torch import nn


# ------------------------------------------------------------------
#  JL random projection matrix  (Gaussian, matching paper)
# ------------------------------------------------------------------
def generate_jl_matrix(input_dim, output_dim, seed=42):
    """Gaussian JL matrix N(0, 1/k)."""
    rng = np.random.RandomState(seed)
    R = rng.normal(0, 1.0 / np.sqrt(output_dim), size=(output_dim, input_dim))
    return R


# ------------------------------------------------------------------
#  Reference gradient helper
# ------------------------------------------------------------------
def compute_reference_gradient(model_class, global_model_np,
                               trusted_loader, dev, loss_func, lr=0.01):
    """
    Compute reference gradient on trusted dataset D_s.

    Matches the FLTrust paper: accumulates gradient over all samples in
    D_s (without weight updates), then takes a SINGLE SGD step.
    The returned delta (θ'_global - θ_global) represents the server's
    trusted update direction.
    """
    net = model_class().to(dev)
    state_dict = net.state_dict()
    for key, param in zip(state_dict.keys(), global_model_np):
        state_dict[key] = torch.from_numpy(param).to(dev)
    net.load_state_dict(state_dict, strict=True)

    opti = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0)

    # Accumulate gradient over ALL samples in D_s
    net.train()
    net.zero_grad()
    total_samples = 0
    for data, label in trusted_loader:
        data, label = data.to(dev), label.to(dev)
        out = net(data)
        loss = loss_func(out, label)
        # Scale by batch size so the accumulated grad = average over all samples
        (loss * len(data)).backward()
        total_samples += len(data)

    # Single SGD step
    for param in net.parameters():
        if param.grad is not None:
            param.grad /= max(total_samples, 1)

    opti.step()

    # delta = θ_after_one_step - θ_before
    delta_parts = []
    new_state = net.state_dict()
    for key, old in zip(new_state.keys(), global_model_np):
        delta_parts.append((new_state[key].cpu().numpy() - old).flatten())
    return np.concatenate(delta_parts)


# ------------------------------------------------------------------
#  Qin et al. aggregation  (single-server, crypto-free for accuracy)
# ------------------------------------------------------------------
def qin_aggregate(client_updates,
                  ref_gradient,
                  jl_matrix=None,
                  eps=1e-8):
    """
    FLTrust-style Byzantine-robust weighted aggregation.

    Parameters
    ----------
    client_updates : list of np.ndarray, each shape (d,)
        Raw model deltas from clients.
    ref_gradient : np.ndarray, shape (d,)
        Trusted reference gradient computed on D_s.
    jl_matrix : np.ndarray shape (k,d) or None
        JL compression matrix.  If None, no compression.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    global_update : np.ndarray, shape (d,)
    weights : np.ndarray, shape (n,)
    """
    n = len(client_updates)
    d = client_updates[0].shape[0]

    G = np.stack(client_updates, axis=0)          # (n, d)
    g_std = ref_gradient.copy()

    # ----- optional JL compression -----
    if jl_matrix is not None:
        G_jl = G @ jl_matrix.T                     # (n, k)
        g_std_jl = jl_matrix @ g_std               # (k,)
    else:
        G_jl = G                                   # (n, d)
        g_std_jl = g_std                           # (d,)

    # ----- norms -----
    norm_std = float(np.linalg.norm(g_std_jl))
    norm_i = np.linalg.norm(G_jl, axis=1)          # (n,)

    # ----- cosine similarity -----
    dot_i = G_jl @ g_std_jl                         # (n,)
    cos_i = dot_i / np.maximum(norm_i, eps) / max(norm_std, eps)

    # ----- trust scores & weights (Eq. 5-6 in the paper) -----
    TS = np.maximum(0.0, cos_i)                     # (n,)
    TS_sum = TS.sum()
    if TS_sum < eps:
        # all distrusted → uniform weights as fallback
        alpha = np.ones(n) / n
    else:
        alpha = TS * (norm_std / np.maximum(norm_i, eps))
        alpha /= alpha.sum()

    # ----- weighted aggregation in FULL dimension -----
    global_update = np.sum(G * alpha[:, np.newaxis], axis=0)   # (d,)

    return global_update, alpha
