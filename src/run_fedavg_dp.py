"""
DP-FedAvg Training Script
==========================
DP-FedAvg: Federated Averaging with client-side differential privacy.
Standard DP-SGD applied to FedAvg: each client clips its model update
to a bounded L2 norm and adds Gaussian noise before uploading.

Reference:
  McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data", AISTATS 2017.

Usage: via run_experiments.py (METHOD = "fedavg_dp")
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from loguru import logger

from common_server import BaseServer
from common_attack import apply_attack


class DPClient:
    """Client that trains locally and adds DP noise (clip + Gaussian)."""

    def __init__(self, cid, train_dl, attack_mode="none", num_classes=43,
                 clip_norm=1.0, noise_std=0.1):
        self.id = cid
        self.train_dl = train_dl
        self.attack_mode = attack_mode
        self.num_classes = num_classes
        self.clip_norm = clip_norm
        self.noise_std = noise_std
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

        # 计算模型更新 delta
        curr = net.state_dict()
        delta = np.concatenate([
            (curr[k].cpu().numpy() - old).flatten()
            for k, old in zip(curr.keys(), global_model)
        ]).astype(np.float64)

        # ---- DP: L2 裁剪 + 高斯噪声 ----
        norm = np.linalg.norm(delta)
        if norm > self.clip_norm:
            delta = delta * (self.clip_norm / norm)
        delta = delta + np.random.normal(0, self.noise_std, size=delta.shape)

        return delta.astype(np.float32)

    def local_val(self, test_dl):
        pass


class FedAvgDPServer(BaseServer):
    """DP-FedAvg: client-side DP noise + plain FedAvg aggregation."""

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 clip_norm=1.0, noise_std=0.1, seed=42):
        super().__init__(num_clients, attack_mode, malicious_ratio,
                         data_distribution, dirichlet_alpha, dataset, seed=seed)
        self.clip_norm = clip_norm
        self.noise_std = noise_std

    # ---- 用 DPClient 替代 VanillaClient ----
    def _make_client(self, idx, images, labels, attack):
        from torch.utils.data import DataLoader, TensorDataset
        dl = DataLoader(TensorDataset(images, labels),
                        batch_size=64, shuffle=True, drop_last=True)
        self.clients[f"client_{idx}"] = DPClient(
            f"client_{idx}", dl, attack,
            num_classes=self.num_classes,
            clip_norm=self.clip_norm, noise_std=self.noise_std)

    def _aggregate(self, deltas):
        return np.mean(deltas, axis=0)
