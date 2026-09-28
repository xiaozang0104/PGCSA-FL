"""
ESFL Training Script
=====================
ESFL: Efficient and Secure Federated Learning Against Backdoor Attacks
(Miao et al., IEEE TDSC 2024).

Core mechanism (Local Differential Privacy + backdoor defense):
  1. Client trains locally (SGD), computes model update delta.
  2. Gaussian noise N(0, sigma^2) added to delta (backdoor mitigation).
  3. Adaptive LDP perturbation: per weight, sample Bernoulli and rescale
     the offset from the layer center by reciprocal factors (ESFL's
     contribution; the "shrink" branch is more likely, partially undoing
     adversarial amplification).
  4. Server aggregates via plain FedAvg.

The DCT + compressive-sensing compression is omitted here (it is an
efficiency mechanism, not a DP/robustness mechanism, and its solver is
unspecified in the paper).

Usage: via run_experiments.py (METHOD = "esfl")
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from common_server import BaseServer
from common_attack import apply_attack


def _adaptive_ldp(delta, epsilon):
    """Adaptive LDP perturbation (ESFL Algorithm 3).

    For each weight: offset mu = w - c_l, then Bernoulli-sample b with
    Pr[b=1] = (e^eps-1)/(2e^eps); if b=1 expand by (e^eps+1)/(e^eps-1),
    else shrink by (e^eps-1)/(e^eps+1).
    """
    e_eps = np.exp(epsilon)
    w_max = delta.max()
    w_min = delta.min()
    c_l = (w_max + w_min) / 2.0          # layer center
    mu = delta - c_l                      # signed offset

    p1 = (e_eps - 1.0) / (2.0 * e_eps)   # Pr[b=1]  (< 0.5)
    b = np.random.random(delta.shape) < p1

    expand = (e_eps + 1.0) / (e_eps - 1.0)   # > 1
    shrink = (e_eps - 1.0) / (e_eps + 1.0)   # < 1
    factor = np.where(b, expand, shrink)

    return c_l + mu * factor


class ESFLClient:
    """Client: local training + Gaussian noise + adaptive LDP perturbation."""

    def __init__(self, cid, train_dl, attack_mode="none", num_classes=43,
                 sigma=0.1, epsilon=2.0):
        self.id = cid
        self.train_dl = train_dl
        self.attack_mode = attack_mode
        self.num_classes = num_classes
        self.sigma = sigma
        self.epsilon = epsilon
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

        # ---- ESFL: 高斯噪声（后门防御）+ 自适应 LDP 扰动 ----
        delta = delta + np.random.normal(0, self.sigma, size=delta.shape)
        delta = _adaptive_ldp(delta, self.epsilon)

        return delta.astype(np.float32)

    def local_val(self, test_dl):
        pass


class ESFLServer(BaseServer):
    """ESFL: Gaussian noise + adaptive LDP perturbation + FedAvg."""

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 sigma=0.1, epsilon=2.0, seed=42):
        super().__init__(num_clients, attack_mode, malicious_ratio,
                         data_distribution, dirichlet_alpha, dataset, seed=seed)
        self.sigma = sigma
        self.epsilon = epsilon

    def _make_client(self, idx, images, labels, attack):
        from torch.utils.data import DataLoader, TensorDataset
        dl = DataLoader(TensorDataset(images, labels),
                        batch_size=64, shuffle=True, drop_last=True)
        self.clients[f"client_{idx}"] = ESFLClient(
            f"client_{idx}", dl, attack,
            num_classes=self.num_classes,
            sigma=self.sigma, epsilon=self.epsilon)

    def _aggregate(self, deltas):
        return np.mean(np.stack(deltas, axis=0), axis=0)
