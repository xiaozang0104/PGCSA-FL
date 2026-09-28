"""
DnC Training Script
===================
Standalone entry point for DnC (Divide-and-Conquer, NDSS 2021).

Usage: via run_experiments.py (METHOD = "dnc")
"""

from common_server import BaseServer
from dnc_aggregator import dnc_aggregate


class DncServer(BaseServer):
    """DnC: Divide-and-Conquer Byzantine-robust FL aggregation."""

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 n_iters=1, c=1, b=10000, seed=42):
        super().__init__(num_clients, attack_mode, malicious_ratio,
                         data_distribution, dirichlet_alpha, dataset, seed=seed)
        self.n_iters = n_iters
        self.c = c
        self.b = b

    def _aggregate(self, deltas):
        m = int(self.num_clients * self.malicious_ratio)
        return dnc_aggregate(deltas, m, n_iters=self.n_iters, c=self.c, b=self.b)
