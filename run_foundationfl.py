"""
FoundationFL Training Script
=============================
Standalone training entry point for FoundationFL (NDSS 2025).

Usage:  python run_foundationfl.py
"""

from loguru import logger

from common_server import BaseServer
from foundationfl_aggregator import foundationfl_aggregate


class FoundationFLServer(BaseServer):
    def __init__(self, *args, ff_rule='trimmed_mean', ff_m=None, ff_c=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff_rule = ff_rule
        self.ff_m = ff_m
        self.ff_c = ff_c

    def _aggregate(self, deltas):
        n = len(deltas)
        return foundationfl_aggregate(deltas,
                                      m=self.ff_m or n // 2,
                                      c=self.ff_c or max(1, n // 5),
                                      rule=self.ff_rule)


# =====================================================================