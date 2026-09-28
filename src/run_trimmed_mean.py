"""
Trimmed Mean / Median Training Script
=======================================
Standalone training entry point for Trimmed Mean and Median aggregation
rules from Fang et al. (USENIX Security 2020).

Usage:  python run_trimmed_mean.py
"""

import numpy as np
from loguru import logger

from common_server import BaseServer
from trimmed_mean_aggregator import trimmed_mean_aggregate, median_aggregate


class TrimmedMeanServer(BaseServer):
    def __init__(self, *args, agg_rule='trimmed_mean', **kwargs):
        super().__init__(*args, **kwargs)
        self.agg_rule = agg_rule  # 'trimmed_mean' or 'median'

    def _aggregate(self, deltas):
        f = int(self.num_clients * self.malicious_ratio)
        if self.agg_rule == 'median':
            return median_aggregate(deltas)
        else:
            return trimmed_mean_aggregate(deltas, f)


# =====================================================================