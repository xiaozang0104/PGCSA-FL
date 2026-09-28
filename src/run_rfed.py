"""
RFed Training Script
=====================
RFed: scaled dot-product attention aggregation (IEEE TIFS 2024).

Server trains a root gradient g0 once on a small clean root dataset,
then scores every client update against g0 each round.

Usage: via run_experiments.py (METHOD = "rfed")
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import optim
from loguru import logger

from common_server import BaseServer
from rfed_aggregator import rfed_aggregate


class RFedServer(BaseServer):
    """RFed: scaled dot-product attention (IEEE TIFS 2024)."""

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 root_size=100, seed=42):
        super().__init__(num_clients, attack_mode, malicious_ratio,
                         data_distribution, dirichlet_alpha, dataset, seed=seed)
        self.root_size = root_size
        self._root_loader = None
        self._g0 = None  # root gradient benchmark (trained once)

    # ------------------------------------------------------------------
    #  Prepare root dataset (same approach as FLTrust)
    # ------------------------------------------------------------------
    def _init_dataset(self):
        super()._init_dataset()
        test_dataset = self.test_data_loader.dataset
        n_test = len(test_dataset)
        size = min(self.root_size * 10, n_test)
        indices = np.random.RandomState(42).choice(n_test, size=size, replace=False)
        root_idx = indices[:self.root_size]
        root_tensors = [test_dataset[i] for i in root_idx]
        root_data = torch.stack([t[0] for t in root_tensors])
        root_label = torch.tensor([t[1] for t in root_tensors], dtype=torch.long)
        self._root_loader = DataLoader(
            TensorDataset(root_data, root_label),
            batch_size=min(64, self.root_size), shuffle=True, drop_last=False,
        )

    # ------------------------------------------------------------------
    #  Train root gradient g0 ONCE on root data (before round 1)
    # ------------------------------------------------------------------
    def _compute_root_gradient(self, model_class, local_epochs):
        net = model_class().to(self.dev)
        opti = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9)
        loss_func = F.cross_entropy

        state_dict = net.state_dict()
        for key, param in zip(state_dict.keys(), self.global_model):
            state_dict[key] = torch.from_numpy(param).to(self.dev)
        net.load_state_dict(state_dict, strict=True)

        for _ in range(local_epochs):
            for data, label in self._root_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                opti.zero_grad()
                loss = loss_func(net(data), label)
                loss.backward()
                opti.step()

        curr = net.state_dict()
        delta = np.concatenate([
            (curr[k].cpu().numpy() - old).flatten()
            for k, old in zip(curr.keys(), self.global_model)
        ])
        return delta

    # ------------------------------------------------------------------
    #  Override run: train g0 once, then standard loop with rfed_aggregate
    # ------------------------------------------------------------------
    def run(self, model_class=None, rounds=100, local_epochs=5, save_csv=None,
            method_name="rfed"):
        logger.info(f"===== {method_name} | {self.dataset} | "
                    f"{self.attack_mode} | {self.data_distribution} =====")
        self._set_seed()

        from common_server import _get_model_class
        if model_class is None:
            model_class = _get_model_class(self.dataset)

        self._init_global_model(model_class)
        self._init_dataset()
        self._init_clients()

        # ---- Train root gradient ONCE ----
        self._g0 = self._compute_root_gradient(model_class, local_epochs)

        if save_csv is None:
            save_csv = (f"results/{method_name}_{self.dataset}_"
                        f"{self.attack_mode}_"
                        f"mal{self.malicious_ratio}_{self.data_distribution}.csv")
        with open(save_csv, 'w') as f:
            f.write("round,accuracy,loss,agg_time_s\n")
        logger.info(f"Results → {save_csv}")

        import time
        loss_func = F.cross_entropy

        for rnd in range(rounds):
            deltas = []
            for cid, client in self.clients.items():
                delta = client.local_train(self.global_model.copy(),
                                           self.lr, local_epochs,
                                           self.model_class)
                deltas.append(delta)

            t0 = time.time()
            agg = rfed_aggregate(deltas, self._g0)
            agg_time = time.time() - t0

            self._update_global_model(agg)

            # 评估（先切 eval 模式，确保 dropout 关闭）
            self.net.eval()
            correct, total = 0, 0
            for data, label in self.test_data_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                correct += (torch.argmax(self.net(data), dim=1) == label).sum().item()
                total += label.size(0)
            acc = correct / max(total, 1)

            self.net.eval()
            total_loss, n_batches = 0.0, 0
            with torch.no_grad():
                for data, label in self.test_data_loader:
                    data, label = data.to(self.dev), label.to(self.dev)
                    total_loss += loss_func(self.net(data), label).item()
                    n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)

            logger.info(f"Round {rnd:3d} | acc: {acc:.4f} | loss: {avg_loss:.4f} | "
                        f"agg_time: {agg_time:.4f}s")

            with open(save_csv, 'a') as f:
                f.write(f"{rnd},{acc:.6f},{avg_loss:.6f},{agg_time:.6f}\n")

        logger.info(f"===== {method_name} training finished | {save_csv} =====")

    # ------------------------------------------------------------------
    #  _aggregate is NOT used — run() calls rfed_aggregate directly
    # ------------------------------------------------------------------
    def _aggregate(self, deltas):
        raise NotImplementedError(
            "RFedServer uses a custom run() that bypasses _aggregate.")
