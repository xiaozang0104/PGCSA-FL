"""
PFLF Training Script
=====================
PFLF: Privacy-Preserving Federated Learning Framework for Edge Computing
(Zhou et al., IEEE TIFS 2022).

Core mechanism (Local Differential Privacy + Gaussian + FedAvg):
  1. Client trains locally with FedProx-style proximal regularization:
       F_i(w) + (mu/2) * ||w - w_global||^2
  2. Client adds Gaussian noise to the model update (LDP).
  3. Server aggregates via plain weighted FedAvg.

No Byzantine-robust aggregation (the paper assumes semi-honest server
and passive eavesdroppers, no malicious clients).

Usage: via run_experiments.py (METHOD = "pflf")
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from common_server import BaseServer
from common_attack import apply_attack


class PFLFClient:
    """Client: FedProx local training + LDP Gaussian noise."""

    def __init__(self, cid, train_dl, attack_mode="none", num_classes=43,
                 mu=0.01, noise_std=0.1):
        self.id = cid
        self.train_dl = train_dl
        self.attack_mode = attack_mode
        self.num_classes = num_classes
        self.mu = mu
        self.noise_std = noise_std
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def local_train(self, global_model, lr, epochs, model_class):
        net = model_class().to(self.dev)
        opti = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        loss_func = F.cross_entropy

        state_dict = net.state_dict()
        global_params = []
        for key, param in zip(state_dict.keys(), global_model):
            p = torch.from_numpy(param).to(self.dev)
            state_dict[key] = p
            global_params.append(p)
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

                # FedProx proximal term: (mu/2) * ||w - w_global||^2
                prox = 0.0
                for param, gp in zip(net.parameters(), global_params):
                    prox += torch.sum((param - gp) ** 2)
                loss = loss + (self.mu / 2.0) * prox

                loss.backward()
                opti.step()

        # 计算 delta
        curr = net.state_dict()
        delta = np.concatenate([
            (curr[k].cpu().numpy() - old).flatten()
            for k, old in zip(curr.keys(), global_model)
        ]).astype(np.float64)

        # ---- LDP: 高斯噪声 ----
        delta = delta + np.random.normal(0, self.noise_std, size=delta.shape)

        return delta.astype(np.float32)

    def local_val(self, test_dl):
        pass


class PFLFServer(BaseServer):
    """PFLF: FedProx + LDP Gaussian noise + FedAvg."""

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 mu=0.01, noise_std=0.1, seed=42):
        super().__init__(num_clients, attack_mode, malicious_ratio,
                         data_distribution, dirichlet_alpha, dataset, seed=seed)
        self.mu = mu
        self.noise_std = noise_std

    def _make_client(self, idx, images, labels, attack):
        from torch.utils.data import DataLoader, TensorDataset
        dl = DataLoader(TensorDataset(images, labels),
                        batch_size=64, shuffle=True, drop_last=True)
        self.clients[f"client_{idx}"] = PFLFClient(
            f"client_{idx}", dl, attack,
            num_classes=self.num_classes,
            mu=self.mu, noise_std=self.noise_std)

    def _aggregate(self, deltas):
        return np.mean(np.stack(deltas, axis=0), axis=0)
