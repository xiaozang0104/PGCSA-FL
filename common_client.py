"""
Common Client Module
=====================
Shared VanillaClient used by all non-PGCSA-FL comparison methods.
No JL compression, no DP noise — just local SGD training.
"""

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from loguru import logger

from common_attack import apply_attack


class VanillaClient:
    """Client that performs plain local SGD and returns the raw model delta."""

    def __init__(self, cid, train_dl, attack_mode="none", num_classes=43):
        self.id = cid
        self.train_dl = train_dl
        self.attack_mode = attack_mode
        self.num_classes = num_classes
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def local_train(self, global_model, lr, epochs, model_class):
        self.net = model_class().to(self.dev)
        self.opti = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        loss_func = F.cross_entropy

        state_dict = self.net.state_dict()
        for key, param in zip(state_dict.keys(), global_model):
            state_dict[key] = torch.from_numpy(param).to(self.dev)
        self.net.load_state_dict(state_dict, strict=True)

        for _ in range(epochs):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                data, label = apply_attack(data, label, self.attack_mode,
                                           self.num_classes,
                                           net=self.net, opti=self.opti,
                                           loss_func=loss_func)
                self.opti.zero_grad()
                loss = loss_func(self.net(data), label)
                loss.backward()
                self.opti.step()

        global_copy = list(global_model)
        curr = self.net.state_dict()
        delta = np.concatenate([(curr[k].cpu().numpy() - old).flatten()
                                for k, old in zip(curr.keys(), global_copy)])
        return delta

    def local_val(self, test_dl):
        self.net.eval()
        correct, total = 0, 0
        for data, label in test_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            correct += (torch.argmax(self.net(data), dim=1) == label).sum().item()
            total += label.size(0)
        logger.info(f"user:{self.id} | local val acc: {correct / max(total, 1):.3f}")
