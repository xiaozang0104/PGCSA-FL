"""
LSFL Training Script
=====================
LSFL: median-of-distances top-k selection (IEEE TIFS 2023).

Usage: via run_experiments.py (METHOD = "lsfl")
"""

from common_server import BaseServer
from lsfl_aggregator import lsfl_aggregate


class LSFLServer(BaseServer):
    """LSFL: L2-distance median baseline + top-k selection."""

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 k_frac=0.7, seed=42):
        super().__init__(num_clients, attack_mode, malicious_ratio,
                         data_distribution, dirichlet_alpha, dataset, seed=seed)
        self.k_frac = k_frac

    def _aggregate(self, deltas):
        return lsfl_aggregate(deltas, k_frac=self.k_frac)
