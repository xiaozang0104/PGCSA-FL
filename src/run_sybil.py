"""
Sybil / FoolsGold Training Script
==================================
FoolsGold: Sybil-resilient FL aggregation (RAID 2020).

Usage: via run_experiments.py (METHOD = "sybil")
"""

from common_server import BaseServer
from sybil_aggregator import sybil_aggregate


class SybilServer(BaseServer):
    """FoolsGold: Sybil-resilient FL aggregation (RAID 2020).

    Maintains cumulative gradient history across rounds for similarity
    computation.  Honest clients' gradients diverge over time; Sybils'
    converge → high cosine similarity → penalised.
    """

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 seed=42):
        super().__init__(num_clients, attack_mode, malicious_ratio,
                         data_distribution, dirichlet_alpha, dataset, seed=seed)
        self._sybil_history = None

    def _aggregate(self, deltas):
        agg, self._sybil_history = sybil_aggregate(deltas, self._sybil_history)
        return agg
