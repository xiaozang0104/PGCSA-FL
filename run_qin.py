"""
Qin et al. (TIFS 2026) Training Script
======================================
Standalone training entry point for the FLTrust-style defense from:
"Efficient Byzantine-Robust Privacy-Preserving Federated Learning
 via Dimension Compression" — Qin, Yang, Tang, IEEE TIFS 2026.

Usage:  python run_qin.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
from loguru import logger

from common_server import BaseServer
from qin_aggregator import (generate_jl_matrix,
                            compute_reference_gradient,
                            qin_aggregate)


class QinServer(BaseServer):
    def __init__(self, *args, trusted_size=100, use_jl=True, reduced_dim=64,
                 jl_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.trusted_size = trusted_size
        self.use_jl = use_jl
        self.reduced_dim = reduced_dim
        self.jl_seed = jl_seed
        self.jl_matrix = None
        self.trusted_loader = None

    def _init_dataset(self):
        super()._init_dataset()

        # ---- build trusted dataset D_s (balanced, from test set) ----
        test_data = self.test_data_loader.dataset.tensors[0]
        test_label = self.test_data_loader.dataset.tensors[1]
        test_labels_np = test_label.numpy()
        unique_classes = np.unique(test_labels_np)
        per_class = self.trusted_size // len(unique_classes)
        trusted_indices = []
        for c in unique_classes:
            idx_c = np.where(test_labels_np == c)[0]
            chosen = np.random.RandomState(self.jl_seed).choice(
                idx_c, size=min(per_class, len(idx_c)), replace=False)
            trusted_indices.extend(chosen.tolist())
        trusted_subset = Subset(TensorDataset(test_data, test_label),
                                 trusted_indices)
        self.trusted_loader = DataLoader(trusted_subset,
                                         batch_size=64, shuffle=True,
                                         drop_last=False)
        logger.info(f"Trusted dataset D_s: {len(trusted_indices)} samples")

        # ---- JL matrix ----
        if self.use_jl:
            self.jl_matrix = generate_jl_matrix(self.input_dim, self.reduced_dim,
                                                seed=self.jl_seed)
            logger.info(f"JL matrix: k={self.reduced_dim}, d={self.input_dim}")

    def _aggregate(self, deltas):
        g_std = compute_reference_gradient(self.model_class,
                                           self.global_model,
                                           self.trusted_loader,
                                           self.dev, F.cross_entropy,
                                           lr=self.lr)
        agg, _ = qin_aggregate(deltas, g_std, jl_matrix=self.jl_matrix)
        return agg


# =====================================================================