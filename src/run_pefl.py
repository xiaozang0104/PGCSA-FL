"""
PEFL Training Script
=====================
Privacy-Enhanced Federated Learning poisoning defense (IEEE TIFS 2021).

Usage: via run_experiments.py (METHOD = "pefl")
"""

from common_server import BaseServer
from pefl_aggregator import pefl_aggregate


class PEFLServer(BaseServer):
    """PEFL: Pearson correlation + Fisher z-transform aggregation."""

    def _aggregate(self, deltas):
        return pefl_aggregate(deltas)
