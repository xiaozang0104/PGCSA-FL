"""
Common Server Module
=====================
Shared base server for all comparison methods.  Handles:
  - global model init
  - dataset loading (GTSRB / MNIST / CIFAR-10)
  - client creation (IID / Dirichlet non‑IID)
  - global model update
  - evaluation + CSV logging
"""

import os
import random
import struct
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.datasets as tv_datasets
from loguru import logger

from dataset import GetDataSet
from common_client import VanillaClient


# ------------------------------------------------------------------
#  Dataset → model map  (imported lazily on first use)
# ------------------------------------------------------------------
_MODEL_CACHE = {}

def _get_model_class(dataset):
    """Return the CNN class for *dataset* ('GTSRB' / 'MNIST' / 'CIFAR10')."""
    if dataset in _MODEL_CACHE:
        return _MODEL_CACHE[dataset]
    if dataset == "GTSRB":
        from model import CNN
    elif dataset == "MNIST":
        from model_MNIST2 import CNN
    elif dataset == "CIFAR10":
        from model_cifar10 import CNN
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    _MODEL_CACHE[dataset] = CNN
    return CNN


class BaseServer:
    """Base class for FL server with common init / dataset / client logic."""

    def __init__(self, num_clients, attack_mode="none", malicious_ratio=0.2,
                 data_distribution="iid", dirichlet_alpha=0.5, dataset="GTSRB",
                 seed=42):
        self.num_clients = num_clients
        self.attack_mode = attack_mode
        self.malicious_ratio = malicious_ratio
        self.data_distribution = data_distribution
        self.dirichlet_alpha = dirichlet_alpha
        self.dataset = dataset          # "GTSRB" | "MNIST" | "CIFAR10"
        self.seed = seed

        self.lr = 0.01
        self.num_classes = None
        self.clients = {}
        self.global_model = []
        self.input_dim = None
        self.model_class = None
        self.dev = None

    def _set_seed(self):
        """Set all random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------
    #  Global model init
    # ------------------------------------------------------------------
    def _init_global_model(self, model_class):
        logger.info("Initialising global model")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_class = model_class
        self.net = model_class().to(self.dev)
        state_dict = self.net.state_dict()
        self.global_model = [p.cpu().numpy() for p in state_dict.values()]
        self.input_dim = int(sum(np.prod(p.shape) for p in state_dict.values()))

    # ------------------------------------------------------------------
    #  Dataset  (GTSRB default; subclasses can override)
    # ------------------------------------------------------------------
    def _init_dataset(self):
        logger.info(f"Initialising dataset: {self.dataset}")
        if self.dataset == "GTSRB":
            self._init_gtsrb()
        elif self.dataset == "MNIST":
            self._init_mnist()
        elif self.dataset == "CIFAR10":
            self._init_cifar10()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def _init_gtsrb(self):
        x_train, x_test, y_train, y_test = GetDataSet('./GTSRB/Train').load_data()
        self.num_classes = y_train.shape[1]
        self.train_data = torch.tensor(x_train, dtype=torch.float32)
        self.train_label = torch.argmax(torch.tensor(y_train, dtype=torch.float32), dim=1)
        self._make_test_loader(torch.tensor(x_test, dtype=torch.float32),
                               torch.argmax(torch.tensor(y_test, dtype=torch.float32), dim=1))

    def _init_mnist(self):
        def _load_images(path):
            with open(path, 'rb') as f:
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                return np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

        def _load_labels(path):
            with open(path, 'rb') as f:
                magic, num = struct.unpack(">II", f.read(8))
                return np.fromfile(f, dtype=np.uint8)

        base = './MNIST'
        x_train = _load_images(os.path.join(base, 'train-images-idx3-ubyte')).astype(np.float32) / 255.0
        y_train = _load_labels(os.path.join(base, 'train-labels-idx1-ubyte')).astype(np.int64)
        x_test  = _load_images(os.path.join(base, 't10k-images-idx3-ubyte')).astype(np.float32) / 255.0
        y_test  = _load_labels(os.path.join(base, 't10k-labels-idx1-ubyte')).astype(np.int64)

        x_train = x_train[:, None, :, :]   # (N,1,28,28)
        x_test  = x_test[:, None, :, :]

        self.num_classes = 10
        self.train_data = torch.tensor(x_train)
        self.train_label = torch.tensor(y_train, dtype=torch.long)
        self._make_test_loader(torch.tensor(x_test),
                               torch.tensor(y_test, dtype=torch.long))

    def _init_cifar10(self):
        try:
            trainset = tv_datasets.CIFAR10(root='.', train=True, download=True)
            testset  = tv_datasets.CIFAR10(root='.', train=False, download=True)
        except Exception:
            trainset = tv_datasets.CIFAR10(root='.', train=True, download=False)
            testset  = tv_datasets.CIFAR10(root='.', train=False, download=False)

        x_train = trainset.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        y_train = np.array(trainset.targets, dtype=np.int64)
        x_test  = testset.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        y_test  = np.array(testset.targets, dtype=np.int64)

        self.num_classes = 10
        self.train_data = torch.tensor(x_train)
        self.train_label = torch.tensor(y_train, dtype=torch.long)
        self._make_test_loader(torch.tensor(x_test),
                               torch.tensor(y_test, dtype=torch.long))

    def _make_test_loader(self, test_data, test_label):
        self.test_data_loader = DataLoader(
            TensorDataset(test_data, test_label),
            batch_size=64, shuffle=False, drop_last=True,
        )

    # ------------------------------------------------------------------
    #  Client distribution
    # ------------------------------------------------------------------
    def _init_clients(self):
        logger.info(f"Initialising {self.data_distribution.upper()} clients")
        N, labels = len(self.train_data), self.train_label.numpy()
        mal_cnt = int(self.num_clients * self.malicious_ratio)

        if self.data_distribution == "iid":
            perm = np.random.permutation(N)  # 打乱，避免数据按标签排序导致 IID 退化为 non-IID
            chunk = N // self.num_clients
            for i in range(self.num_clients):
                idx = perm[chunk * i: chunk * (i + 1)]
                attack = self.attack_mode if i < mal_cnt else "none"
                self._make_client(i, self.train_data[idx], self.train_label[idx], attack)
        else:
            num_classes = labels.max() + 1
            class_indices = {c: np.where(labels == c)[0].tolist()
                             for c in range(num_classes)}
            client_indices = {i: [] for i in range(self.num_clients)}
            for c, idxs in class_indices.items():
                np.random.shuffle(idxs)
                p = np.random.dirichlet([self.dirichlet_alpha] * self.num_clients)
                counts = (p * len(idxs)).astype(int)
                diff = len(idxs) - counts.sum()
                for j in range(diff):
                    counts[j % self.num_clients] += 1
                ptr = 0
                for i in range(self.num_clients):
                    cnt = counts[i]
                    client_indices[i].extend(idxs[ptr:ptr + cnt])
                    ptr += cnt
            for i in range(self.num_clients):
                idx = client_indices[i]
                np.random.shuffle(idx)
                attack = self.attack_mode if i < mal_cnt else "none"
                self._make_client(i, self.train_data[idx], self.train_label[idx], attack)

    def _make_client(self, idx, images, labels, attack):
        dl = DataLoader(TensorDataset(images, labels),
                        batch_size=64, shuffle=True, drop_last=True)
        self.clients[f"client_{idx}"] = VanillaClient(
            f"client_{idx}", dl, attack,
            num_classes=self.num_classes)

    # ------------------------------------------------------------------
    #  Global model update
    # ------------------------------------------------------------------
    def _update_global_model(self, aggregated_delta):
        new_state, new_global = {}, []
        state_dict = self.net.state_dict()
        ptr = 0
        for key, param in state_dict.items():
            sz = int(np.prod(param.shape))
            block = aggregated_delta[ptr:ptr + sz].reshape(param.shape)
            new_state[key] = torch.from_numpy(param.cpu().numpy() + block).to(self.dev)
            new_global.append(param.cpu().numpy() + block)
            ptr += sz
        self.net.load_state_dict(new_state, strict=True)
        self.global_model = new_global

    # ==================================================================
    #  Main loop  (subclass MUST implement _aggregate(deltas) → agg_delta)
    # ==================================================================
    def run(self, model_class=None, rounds=100, local_epochs=5, save_csv=None,
            method_name="method"):
        """Run the FL training loop.

        If *model_class* is None it is auto-selected from ``self.dataset``.
        """
        logger.info(f"===== {method_name} | {self.dataset} | "
                    f"{self.attack_mode} | {self.data_distribution} =====")
        self._set_seed()
        if model_class is None:
            model_class = _get_model_class(self.dataset)
        self._init_global_model(model_class)
        self._init_dataset()
        self._init_clients()

        if save_csv is None:
            save_csv = (f"results/{method_name}_{self.dataset}_"
                        f"{self.attack_mode}_"
                        f"mal{self.malicious_ratio}_{self.data_distribution}.csv")
        with open(save_csv, 'w') as f:
            f.write("round,accuracy,loss,agg_time_s\n")
        logger.info(f"Results → {save_csv}")

        loss_func = F.cross_entropy

        for rnd in range(rounds):
            deltas = []
            for cid, client in self.clients.items():
                delta = client.local_train(self.global_model.copy(),
                                           self.lr, local_epochs,
                                           self.model_class)
                deltas.append(delta)

            # method-specific aggregation (subclass override)
            t0 = time.time()
            agg = self._aggregate(deltas)
            agg_time = time.time() - t0

            self._update_global_model(agg)

            # evaluate（先切 eval 模式，确保 F.dropout(training=...) 关闭）
            self.net.eval()
            correct, total = 0, 0
            for data, label in self.test_data_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                correct += (torch.argmax(self.net(data), dim=1) == label).sum().item()
                total += label.size(0)
            acc = correct / max(total, 1)

            self.net.eval()
            total_loss, n_batches = 0.0, 0
            with torch.no_grad():
                for data, label in self.test_data_loader:
                    data, label = data.to(self.dev), label.to(self.dev)
                    total_loss += loss_func(self.net(data), label).item()
                    n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)

            logger.info(f"Round {rnd:3d} | acc: {acc:.4f} | loss: {avg_loss:.4f} | "
                        f"agg_time: {agg_time:.4f}s")

            with open(save_csv, 'a') as f:
                f.write(f"{rnd},{acc:.6f},{avg_loss:.6f},{agg_time:.6f}\n")

        logger.info(f"===== {method_name} training finished | {save_csv} =====")

    # ----------------------------------------------------------------
    #  Subclass MUST override this
    # ----------------------------------------------------------------
    def _aggregate(self, deltas):
        raise NotImplementedError("Subclass must implement _aggregate(deltas)")
