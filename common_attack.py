"""
Common Attack Module
=====================
Shared attack implementations used by all comparison methods.
All attacks operate on (data, label) and return (data, label).

Paper descriptions:
  Label Flip:     flip 100% of class c → target class c*
  Backdoor:       5% images get 5×5 white trigger lower-right → target c*
  U-Adaptive:     each sample gets a random wrong label from SL\\{c}
  T-Adaptive:     try ≤10 candidate targets per mini-batch, pick the one
                  with max gradient norm, flip all labels in that batch
"""

import numpy as np
import torch


def apply_attack(data, label, attack_mode, num_classes,
                 net=None, opti=None, loss_func=None):
    """
    Apply poisoning attack to a mini-batch.

    Parameters
    ----------
    data : Tensor (B, C, H, W)
    label : Tensor (B,)
    attack_mode : str — "none", "label_flip", "backdoor",
                        "untargeted_adaptive", "targeted_adaptive"
    num_classes : int
    net : nn.Module or None (required for targeted_adaptive)
    opti : Optimizer or None (required for targeted_adaptive)
    loss_func : callable or None (required for targeted_adaptive)

    Returns
    -------
    data, label : (Tensor, Tensor) — possibly modified
    """
    if attack_mode == "none":
        return data, label

    # ---- Label Flip ----
    if attack_mode == "label_flip":
        src_cls, tgt_cls = 1, 0
        mask = (label == src_cls)
        label = torch.where(mask,
                            torch.full_like(label, tgt_cls),
                            label)
        return data, label

    # ---- Backdoor ----
    if attack_mode == "backdoor":
        _, _, H, W = data.shape
        tgt_cls = 0
        poison_mask = (torch.rand(len(data),
                         device=data.device) < 0.05)
        if poison_mask.any():
            data = data.clone()
            data[poison_mask, :, H - 5:H, -5:] = 1.0
            label = label.clone()
            label[poison_mask] = tgt_cls
        return data, label

    # ---- Untargeted Adaptive ----
    if attack_mode == "untargeted_adaptive":
        new_label = label.clone()
        for idx in range(len(label)):
            c = label[idx].item()
            candidates = [x for x in range(num_classes) if x != c]
            new_label[idx] = candidates[np.random.randint(len(candidates))]
        return data, new_label

    # ---- Targeted Adaptive ----
    if attack_mode == "targeted_adaptive":
        orig_label = label.clone()
        max_norm = -1.0
        worst_c = 0
        candidates = list(range(num_classes))
        np.random.shuffle(candidates)
        num_try = min(num_classes, 10)
        for c_try in candidates[:num_try]:
            if (orig_label == c_try).all():
                continue
            opti.zero_grad()
            label_try = torch.full_like(orig_label, c_try)
            loss_try = loss_func(net(data), label_try)
            loss_try.backward()
            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in net.parameters()
                if p.grad is not None
            ) ** 0.5
            opti.zero_grad()
            if grad_norm > max_norm:
                max_norm = grad_norm
                worst_c = c_try
        return data, torch.full_like(orig_label, worst_c)

    return data, label
