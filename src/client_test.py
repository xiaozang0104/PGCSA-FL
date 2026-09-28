# client_test.py

import torch
from torch import optim
import numpy as np
import torch.nn.functional as F
from loguru import logger
from scipy.sparse import random as sparse_random

from common_attack import apply_attack


def generate_jl_matrix(d, k, rng=None):
    """生成稀疏 JL 投影矩阵 $A\\in R^{k\\times d}$。

    :param rng: 随机源; 传入独立的 np.random.RandomState 可避免扰动全局 RNG,
                从而不改变数据划分与后续所有随机性。
    """
    rng = np.random if rng is None else rng
    p = 1.0 / np.sqrt(k)
    A = sparse_random(k, d, density=p,
                      data_rvs=lambda s: rng.choice([1, -1], size=s)).toarray()
    A *= np.sqrt(1.0 / p)
    return A


class BatchClient(object):
    def __init__(self, cid, train_dl, input_dim, reduced_dim, noise_std,
                 attack_mode="none", num_classes=43, model_class=None,
                 jl_matrix=None):
        """
        :param attack_mode: "none", "label_flip", "backdoor",
                            "untargeted_adaptive", "targeted_adaptive"
        :param num_classes: number of classes in the dataset
        :param model_class: CNN class for this dataset (required)
        :param jl_matrix: 服务器下发的共享 JL 矩阵; None 时自行生成(向后兼容)
        """
        self.net = None
        self.opti = None
        self.loss_func = None
        self.dev = None
        self.id = cid
        self.train_dl = train_dl
        self.input_dim = input_dim
        self.reduced_dim = reduced_dim
        self.noise_std = noise_std
        self.attack_mode = attack_mode
        self.num_classes = num_classes
        self.model_class = model_class

        # JL 投影矩阵: 优先使用服务器下发的共享矩阵(论文 scheme.tex:39:
        # "the server initializes A and distributes it to all clients"),
        # 未提供时退回"每个客户端各自生成"的旧行为。
        self.jl_matrix = (jl_matrix if jl_matrix is not None
                          else generate_jl_matrix(self.input_dim, self.reduced_dim))
        # 投影历史（供 IPCA 使用）
        self.history_proj = []

    def generate_jl_matrix(self, d, k):
        """向后兼容的薄封装; 实际逻辑见模块级 generate_jl_matrix。"""
        return generate_jl_matrix(d, k)

    def estimate_principal_subspace(self):
        """基于 history_proj 做 SVD，返回 k×r 的子空间基 U_est"""
        G = np.stack(self.history_proj, axis=1)  # k×T
        U, _, _ = np.linalg.svd(G, full_matrices=False)
        r = min(self.reduced_dim // 10, U.shape[1])
        return U[:, :r]

    def estimate_mean_update(self):
        """基于 history_proj 做均值并映回原始空间，返回 d 维向量"""
        z_bar = np.mean(self.history_proj, axis=0)  # k,
        A_pinv = np.linalg.pinv(self.jl_matrix)     # d×k
        return A_pinv @ z_bar                        # d,

    def local_train(self, global_model, lr, epoch):
        # —— 本地模型训练 —— #
        if self.model_class is None:
            raise RuntimeError(
                "BatchClient.model_class is None. "
                "Pass model_class=CNN when creating BatchClient.")
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = self.model_class().to(self.dev)
        self.loss_func = F.cross_entropy
        self.opti = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        self.net = net

        # 加载全局参数
        state_dict = self.net.state_dict()
        for key, param in zip(state_dict.keys(), global_model):
            state_dict[key] = torch.from_numpy(param).to(self.dev)
        self.net.load_state_dict(state_dict, strict=True)

        # 本地训练（攻击在每batch内执行，匹配论文描述）
        for _ in range(epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)

                data, label = apply_attack(data, label, self.attack_mode,
                                           self.num_classes,
                                           net=self.net, opti=self.opti,
                                           loss_func=self.loss_func)

                self.opti.zero_grad()
                preds = self.net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                self.opti.step()

        # —— 计算原始更新向量 Δ_i —— #
        global_copy = list(global_model)
        updates = []
        curr_state = self.net.state_dict()
        for key, old in zip(curr_state.keys(), global_copy):
            upd = (curr_state[key].cpu().numpy() - old).flatten()
            updates.append(upd)
        delta = np.concatenate(updates)  # (d,)

        # —— JL投影 + 加噪 —— #
        proj = self.jl_matrix @ delta
        noise = np.random.normal(0, self.noise_std, size=proj.shape)
        proj_noisy = proj + noise

        # 更新投影历史（供 IPCA 异常检测使用）
        self.history_proj.append(proj_noisy)
        if len(self.history_proj) > self.reduced_dim * 2:
            self.history_proj.pop(0)

        return delta, proj_noisy

    def local_val(self, test_dl):
        self.net.eval()
        sum_accu = 0; num = 0
        for data, label in test_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = torch.argmax(self.net(data), dim=1)
            sum_accu += (preds == label).float().mean(); num += 1
        logger.info(f'user:{self.id} | local val acc: {sum_accu/num:.3f}')
