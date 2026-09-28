"""
DPDLDA (FL adaptation) Training Script
========================================
DPDLDA: Differentially Private Deep Learning with Dynamic Privacy Budget
Allocation and Adaptive Optimization (Chen et al., IEEE TIFS 2023).

Original paper is a centralized DP-SGD variant.  For FL comparison we
port its two core DP mechanisms to the client side:
  1. Grouped clipping (GroupClip): split the model update into groups,
     clip each group's L2 norm to C.
  2. Sinh-Normal noise (not Gaussian): noise = arsinh((1/eta) * N(0, sigma^2/2)).
Then the server aggregates via plain FedAvg.

(The LRP budget allocation and NoisyMin step-size selection are
centralized-training mechanisms and are omitted here.)

Usage: via run_experiments.py (METHOD = "dpdlda")
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from common_server import BaseServer
from common_attack import apply_attack


def _group_clip(delta, C, n_groups):
    """GroupClip: split delta into n_groups groups, L2-clip each to C."""
    d = len(delta)
    clipped = delta.copy()
    group_size = d // n_groups
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else d
        group = delta[start:end]
        norm = np.linalg.norm(group)
        if norm > C:
            clipped[start:end] = group * (C / norm)
    return clipped


def _sinh_normal_noise(shape, eta, sigma):
    """Sinh-Normal noise: arsinh((1/eta) * N(0, sigma^2/2))."""
    z = np.random.normal(0, np.sqrt(sigma ** 2 / 2), size=shape)
    return np.arcsinh(z / eta)


class DPDLDAClient:
    """Client: local training + grouped clipping + Sinh-Normal noise."""

    def __init__(self, cid, train_dl, attack_mode="none", num_classes=43,
                 clip_norm=4.0, n_groups=8, eta=0.5, sigma=0.1):
        self.id = cid
        self.train_dl = train_dl
        self.attack_mode = attack_mode
        self.num_classes = num_classes
        self.clip_norm = clip_norm
        self.n_groups = n_groups
        self.eta = eta
        self.sigma = sigma
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def local_train(self, global_model, lr, epochs, model_class):
        net = model_class().to(self.dev)
        opti = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        loss_func = F.cross_entropy

        state_dict = net.state_dict()
        for key, param in zip(state_dict.keys(), global_model):
            state_dict[key] = torch.from_numpy(param).to(self.dev)
        net.load_state_dict(state_dict, strict=True)

        for _ in range(epochs):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                data, label = apply_attack(data, label, self.attack_mode,
                                           self.num_classes,
                                           net=net, opti=opti,
                                           loss_func=loss_func)
                opti.zero_grad()
                loss = loss_func(net(data), label)
                loss.backward()
                opti.step()

        # 计算 delta
        curr = net.state_dict()
        delta = np.concatenate([
            (curr[k].cpu().numpy() - old).flatten()
            for k, old in zip(curr.keys(), global_model)
        ]).astype(np.float64)

        # ---- DPDLDA: 分组裁剪 + Sinh-Normal 噪声 ----
        delta = _group_clip(delta, self.clip_norm, self.n_groups)
        delta = delta + _sinh_normal_noise(delta.shape, self.eta, self.sigma)

        return delta.astype(np.float32)

    def local_val(self, test_dl):
        pass


class DPDLDAServer(BaseServer):
    """DPDLDA (FL adaptation): grouped clipping + Sinh-Normal noise + FedAvg."""

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 clip_norm=4.0, n_groups=8, eta=0.5, sigma=0.1, seed=42):
        super().__init__(num_clients, attack_mode, malicious_ratio,
                         data_distribution, dirichlet_alpha, dataset, seed=seed)
        self.clip_norm = clip_norm
        self.n_groups = n_groups
        self.eta = eta
        self.sigma = sigma

    def _make_client(self, idx, images, labels, attack):
        from torch.utils.data import DataLoader, TensorDataset
        dl = DataLoader(TensorDataset(images, labels),
                        batch_size=64, shuffle=True, drop_last=True)
        self.clients[f"client_{idx}"] = DPDLDAClient(
            f"client_{idx}", dl, attack,
            num_classes=self.num_classes,
            clip_norm=self.clip_norm, n_groups=self.n_groups,
            eta=self.eta, sigma=self.sigma)

    def _aggregate(self, deltas):
        return np.mean(np.stack(deltas, axis=0), axis=0)
