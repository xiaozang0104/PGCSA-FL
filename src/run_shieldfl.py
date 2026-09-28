"""
ShieldFL Training Script
=========================
ShieldFL: poisoning defense via cosine-similarity reweighting (IEEE TIFS 2022).

Usage: via run_experiments.py (METHOD = "shieldfl")
"""

from common_server import BaseServer
from shieldfl_aggregator import shieldfl_aggregate


class ShieldFLServer(BaseServer):
    """ShieldFL: cosine-similarity soft reweighting (IEEE TIFS 2022).

    Maintains the previous round's aggregate across rounds to serve as the
    reference for baseline selection.
    """

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 seed=42):
        super().__init__(num_clients, attack_mode, malicious_ratio,
                         data_distribution, dirichlet_alpha, dataset, seed=seed)
        self._prev_agg = None

    def _aggregate(self, deltas):
        agg, self._prev_agg = shieldfl_aggregate(deltas, self._prev_agg)
        return agg
