"""
Krum Training Script
=====================
Standalone training entry point for Krum / Multi-Krum (NIPS 2017).

Usage:  python run_krum.py
"""

import numpy as np
from loguru import logger

from common_server import BaseServer
from krum_aggregator import krum_aggregate


class KrumServer(BaseServer):
    def _aggregate(self, deltas):
        f = int(self.num_clients * self.malicious_ratio)
        return krum_aggregate(deltas, f, multi=True)

