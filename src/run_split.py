"""
Split Aggregation Training Script
==================================
Split Aggregation: SVD projection + median threshold + tanh weighting
(IEEE TIFS 2024).

Usage: via run_experiments.py (METHOD = "split")
"""

from common_server import BaseServer
from split_aggregator import split_aggregate


class SplitServer(BaseServer):
    """Split Aggregation: spectral outlier detection + soft weighting."""

    def _aggregate(self, deltas):
        return split_aggregate(deltas)
