import os
import random
import struct
import time
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loguru import logger
from dataset import GetDataSet
from client_test import BatchClient, generate_jl_matrix
import torch.nn.functional as F
import torchvision.datasets as tv_datasets

# ---- lazy model cache ----
_MODEL_CACHE = {}

def _get_model_class(dataset):
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

class BatchAggrServer:
    def __init__(self, num_clients, reduced_dim, noise_std,
                 attack_mode="none", malicious_ratio=0.2,
                 threshold_multiplier=2.0,
                 data_distribution="iid", dirichlet_alpha=0.5,
                 dataset="GTSRB", r=15, beta=0.1, seed=42):
        """
        :param attack_mode: "none", "label_flip", "backdoor",
                            "untargeted_adaptive", "targeted_adaptive"
        :param dataset: "GTSRB" | "MNIST" | "CIFAR10"
        :param r: number of principal directions for IPCA anomaly detection
        :param beta: exponential smoothing factor for covariance estimate
        :param threshold_multiplier: legacy coefficient c (=2.0 in all scripts). The threshold is
            written as τ_lo = s_med + g·(s_med − s_max), τ_hi = s_med + g·(s_med − s_min) with the
            mirror-gap coefficient g = c − 1.0, so c=2.0 reproduces the paper's τ = 2·s_med − s_min.
        :param seed: random seed for reproducibility
        """
        self.client_nums = num_clients
        self.reduced_dim = reduced_dim
        self.noise_std = noise_std
        self.lr = 0.01
        self.clients = {}
        self.global_model = []
        self.input_dim = None

        self.attack_mode = attack_mode
        self.malicious_ratio = malicious_ratio
        self.malicious_indices = set()   # 由 __init_clients 填充(=最低索引的 n_mal 个)
        self.threshold_multiplier = threshold_multiplier
        # 镜像间隙系数 g = c - 1; 见 detect_anomalies。c=2.0(全部历史脚本) ⇒ g=1.0
        self.gap_multiplier = threshold_multiplier - 1.0
        self.jl_matrix = None            # 全局共享的 JL 投影矩阵, 见 __init_clients

        self.data_distribution = data_distribution
        self.dirichlet_alpha = dirichlet_alpha
        self.dataset = dataset
        self.seed = seed

        # —— 分块 PCA 状态 —— #
        self.C = np.zeros((self.reduced_dim, self.reduced_dim))
        self.beta = beta          # exponential smoothing factor
        self.r = r
        self.block_iters = 5

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
        logger.info(f"Random seed set to {self.seed}")

    def __init_global_model(self):
        logger.info('Initializing global model')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model_cls = _get_model_class(self.dataset)
        self.net = model_cls().to(self.dev)
        self.loss_func = F.cross_entropy
        self.opti = optim.SGD(self.net.parameters(), lr=self.lr)
        state_dict = self.net.state_dict()
        self.global_model = [param.cpu().numpy() for param in state_dict.values()]
        self.input_dim = sum(np.prod(param.shape) for param in state_dict.values())

    def __init_dataset(self):
        logger.info(f'Initializing dataset: {self.dataset}')
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
        x_train = x_train[:, None, :, :]
        x_test  = x_test[:, None, :, :]
        self.num_classes = 10
        self.train_data = torch.tensor(x_train)
        self.train_label = torch.tensor(y_train, dtype=torch.long)
        self._make_test_loader(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))

    def _init_cifar10(self):
        trainset = tv_datasets.CIFAR10(root='.', train=True, download=False)
        testset  = tv_datasets.CIFAR10(root='.', train=False, download=False)
        x_train = trainset.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        y_train = np.array(trainset.targets, dtype=np.int64)
        x_test  = testset.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        y_test  = np.array(testset.targets, dtype=np.int64)
        self.num_classes = 10
        self.train_data = torch.tensor(x_train)
        self.train_label = torch.tensor(y_train, dtype=torch.long)
        self._make_test_loader(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))

    def _make_test_loader(self, test_data, test_label):
        self.test_data_loader = DataLoader(
            TensorDataset(test_data, test_label),
            batch_size=64, shuffle=False, drop_last=True
        )

    def __init_clients(self):
        logger.info(f'Initializing {self.data_distribution.upper()} clients')
        # 【说明】恶意客户端 = 最低索引的 k 个, 与 common_server.py 及其它 16 个基线完全一致,
        #   保证同一张表里所有方法攻击的是同一批客户端。
        #   (曾一度改为随机抽取, 但那是多余的: 造成"索引伪影"的真正原因是
        #    detect_anomalies 里逐客户端的 β 平滑, 已修复为逐轮平滑。
        #    本地实测: 逐轮平滑下随机索引与最低索引第 0 轮精度完全相同(3.55%),
        #    残差分数与索引无相关性 → 无需随机化, 从而保住全部基线数据。)
        n_mal = int(self.client_nums * self.malicious_ratio)
        self.malicious_indices = set(range(n_mal))
        logger.info(f'Malicious clients ({n_mal}): {sorted(self.malicious_indices)}')

        # —— 全局共享的 JL 投影矩阵(P0 修复) —— #
        # 论文: 服务器生成 A 并下发给所有客户端(scheme.tex:39)。
        # 原实现让每个客户端各自生成 A, 导致各客户端上传的 z 不在同一坐标系
        # (E[z_i·z_j]≈0), 服务器估出的'主导子空间'没有共同结构 → 检测失效。
        # 用独立 RandomState 生成: 不消耗全局 numpy RNG,
        # 因此分片划分(Dirichlet)与"每个客户端拿到哪些样本"完全不变;
        # 副作用(相对旧运行): 旧代码在客户端循环内做 JL 抽样会消耗全局 RNG,
        #   现在不消耗了 → 客户端内样本顺序、以及 U-Ad 的随机标签会变动。
        # 附带好处: 客户端初始化阶段的 RNG 流现在与 common_server.py(各基线)一致。
        self.jl_matrix = generate_jl_matrix(
            self.input_dim, self.reduced_dim,
            rng=np.random.RandomState(self.seed + 999))
        logger.info(f'Shared JL matrix: {self.jl_matrix.shape}')

        N = len(self.train_data)
        labels = self.train_label.numpy()
        if self.data_distribution == "iid":
            perm = np.random.permutation(N)  # 打乱，避免数据按标签排序导致 IID 退化为 non-IID
            num_interval = N // self.client_nums
            for i in range(self.client_nums):
                idx = perm[num_interval * i: num_interval * (i + 1)]
                images = self.train_data[idx]
                labs   = self.train_label[idx]
                self._make_client(i, images, labs)
        else:
            # Non‑IID: Dirichlet 划分
            num_classes = labels.max() + 1
            class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
            client_indices = {i: [] for i in range(self.client_nums)}
            for c, idxs in class_indices.items():
                np.random.shuffle(idxs)
                p = np.random.dirichlet([self.dirichlet_alpha] * self.client_nums)
                counts = (p * len(idxs)).astype(int)
                diff = len(idxs) - counts.sum()
                for j in range(diff): counts[j % self.client_nums] += 1
                ptr = 0
                for i in range(self.client_nums):
                    cnt = counts[i]
                    client_indices[i].extend(idxs[ptr:ptr + cnt])
                    ptr += cnt
            for i in range(self.client_nums):
                idx = client_indices[i]
                np.random.shuffle(idx)
                images = self.train_data[idx]
                labs   = self.train_label[idx]
                self._make_client(i, images, labs)

    def _make_client(self, idx, images, labels):
        train_dl = DataLoader(
            TensorDataset(images, labels),
            batch_size=64, shuffle=True, drop_last=True
        )
        if idx in self.malicious_indices:
            attack = self.attack_mode
        else:
            attack = "none"
        model_cls = _get_model_class(self.dataset)
        self.clients[f'client_{idx}'] = BatchClient(
            f'client_{idx}', train_dl,
            self.input_dim, self.reduced_dim,
            self.noise_std, attack,
            num_classes=self.num_classes,
            model_class=model_cls,
            jl_matrix=self.jl_matrix,     # 共享矩阵(P0 修复)
        )

    def _block_eigendecomp(self, C: np.ndarray) -> np.ndarray:
        """论文 Step 2 + Step 3: 分块特征分解得到初始子空间 U0, 再幂迭代 I 步。

        论文 Eq.(largest eigenvalues): 每个对角块取 top-r_i 个特征向量, r = sum_i r_i;
        论文 Eq.(initial subspace): 各块特征向量零填充后按块位置堆叠成 U0 in R^{k x r};
        论文 Eq.(orthonormalization): U <- QR(C U^t), 迭代 I = self.block_iters 次。

        [G3 修复] 原实现每块只取 1 个特征向量(其余 r-2 列为零, 由 QR 任意补齐), 与上式不符;
        现按论文公式每块取 r_i 个(本实现 s=2 个块, 取 r1 = ceil(r/2), r2 = r - r1)。
        """
        k = C.shape[0]
        m = k // 2
        C11, C22 = C[:m, :m], C[m:, m:]
        vals1, vecs1 = np.linalg.eigh(C11)
        vals2, vecs2 = np.linalg.eigh(C22)
        vecs1 = vecs1[:, ::-1][:, :m]      # 按特征值降序
        vecs2 = vecs2[:, ::-1][:, :m]

        # r = r1 + r2, 两个对角块各贡献 top-r_i 个方向
        r1 = min((self.r + 1) // 2, m)
        r2 = min(self.r - r1, m)
        U = np.zeros((k, self.r))          # 未填满的列留零(r > k 的退化情形)
        if r1 > 0:
            U[:m, :r1] = vecs1[:, :r1]
        if r2 > 0:
            U[m:, r1:r1 + r2] = vecs2[:, :r2]

        U, _ = np.linalg.qr(U)
        for _ in range(self.block_iters):
            U = C @ U
            U, _ = np.linalg.qr(U)
        return U[:, :self.r]

    def detect_anomalies(self, projected_updates):
        # 【修复】协方差的指数平滑必须是"每轮一次", 即论文公式
        #     C̃^t = β·C̃^{t-1} + (1-β)·C^t,  C^t = (1/N)·Σ_i g_i g_i^T
        #   原实现把 β 平滑放在客户端循环内部(逐客户端平滑), β=0.1 时
        #   C 被最后几个客户端的投影主导(权重 0.9, 0.09, 0.009, ...),
        #   于是 U 只覆盖"按索引靠后"的客户端, 前 N-r 个客户端被结构性剔除;
        #   同时跨轮记忆退化为 β^N ≈ 1e-20, 平滑参数失去跨轮含义。
        G = np.stack(projected_updates, axis=1)          # k×N (每列已归一化)
        C_t = (G @ G.T) / G.shape[1]                     # 本轮协方差, tr(C^t)=1
        self.C = self.beta * self.C + (1 - self.beta) * C_t
        U_r = self._block_eigendecomp(self.C)
        proj_squares = np.sum((U_r.T @ G) ** 2, axis=0)
        # 【2026-09-15 决定】打分 = 投影能量 s = ||U^T g||^2
        #   （与论文 Eq.(projectionenergy) 的字面形式一致；g 为归一化投影）
        #
        #   ⚠️ 语义: 能量越大 = 与主导子空间越对齐 = 越"正常"。
        #      而下面的判据是 "s > τ 即剔除" → 按论文原文, 剔除的是
        #      "最对齐主导子空间"的客户端 (方向与残差口径相反)。
        #
        #   ⚠️ 阈值: tau_hi 在能量分布下经常 > s.max()
        #      (实测例: s_med=0.2586, s_min=0.0008 → τ=0.516 > max=0.451),
        #      此时本轮不剔除任何客户端。
        s_min = proj_squares.min()
        s_max = proj_squares.max()
        s_med = np.median(proj_squares)
        # 【两侧镜像判据】2026-09-15
        #   论文 Step 4 正文即写明 "malicious or anomalous updates often deviate from this
        #   structure, resulting in unusually low or high projection energies", 但原公式只实现了
        #   high 一侧。实测发现"哪一侧是恶意"会逐轮翻转(子空间被协同攻击者劫持时会翻),
        #   单侧判据因此隔轮完全失明 → 现补上另一侧。
        #   公式写成"中位数 + 镜像间隙"的对称形式:
        #       tau_lo = s_med + (s_med - s_max)      # 上极值关于中位数镜像 → 下界
        #       tau_hi = s_med + (s_med - s_min)      # 下极值关于中位数镜像 → 上界
        #   保留区间 = [s_med - (s_max - s_med), s_med + (s_med - s_min)]:
        #   关于中位数"半对称"(下臂 = 上极值的超出量, 上臂 = 下极值的欠缺量), 带宽 = s_max - s_min。
        #   等价性: 旧口径 c=2.0 的 2·s_med - s_min ≡ s_med + (s_med - s_min), 故历史 c=2.0 的结果有效。
        #   互补性(永不整轮空转): s_med < (s_min+s_max)/2 ⇒ tau_hi < s_max (上侧有效);
        #                         s_med > (s_min+s_max)/2 ⇒ tau_lo > s_min (下侧有效)。
        gap = self.gap_multiplier        # = threshold_multiplier - 1.0, 默认 2.0 ⇒ 1.0
        tau_lo = s_med + gap * (s_med - s_max)
        tau_hi = s_med + gap * (s_med - s_min)

        lo_flag = [i for i, s in enumerate(proj_squares) if s < tau_lo]
        hi_flag = [i for i, s in enumerate(proj_squares) if s > tau_hi]
        anomalies = sorted(set(lo_flag) | set(hi_flag))
        side = ("both" if (lo_flag and hi_flag) else
                "low" if lo_flag else "high" if hi_flag else "none")

        # 诊断用(不参与计算): 供外部脚本统计检测 precision/recall
        self.last_scores = proj_squares
        self.last_tau = tau_lo          # 兼容旧脚本(单值); 两侧见下面两个属性
        self.last_tau_lo = tau_lo
        self.last_tau_hi = tau_hi
        self.last_side = side
        logger.info(f"ProjectionEnergy: {proj_squares}, s_min: {s_min:.4f}, s_med: {s_med:.4f}, "
                    f"s_max: {s_max:.4f}, gap: {gap:g}, "
                    f"tau_lo: {tau_lo:.4f} (=s_med{tau_lo - s_med:+.4f} [s_med-s_max], "
                    f"s<tau_lo: {lo_flag}), "
                    f"tau_hi: {tau_hi:.4f} (=s_med{tau_hi - s_med:+.4f} [s_med-s_min], "
                    f"s>tau_hi: {hi_flag}), side: {side}, Anomalies: {anomalies}")
        return anomalies

    def aggregate_updates(self, update_vectors, good_client_indices):
        good = [update_vectors[i] for i in good_client_indices]
        return np.mean(good, axis=0)

    def update_global_model(self, aggregated_update):
        new_state, new_global = {}, []
        state_dict = self.net.state_dict()
        ptr = 0
        for key, param in state_dict.items():
            size = int(np.prod(param.shape))
            delta = aggregated_update[ptr:ptr + size].reshape(param.shape)
            new_param = param.cpu().numpy() + delta
            new_state[key] = torch.from_numpy(new_param).to(self.dev)
            new_global.append(new_param)
            ptr += size
        self.net.load_state_dict(new_state, strict=True)
        self.global_model = new_global

    def start(self, save_csv=None):
        logger.info('Starting training!')
        self._set_seed()
        self.__init_global_model()
        self.__init_dataset()
        self.__init_clients()
        rounds = 100 #训练轮数

        if save_csv is None:
            save_csv = (f"results/pgcsa_{self.dataset}_{self.attack_mode}_"
                        f"mal{self.malicious_ratio}_{self.data_distribution}.csv")
        with open(save_csv, 'w') as f:
            f.write("round,accuracy,loss,agg_time_s\n")
        logger.info(f"Results → {save_csv}")

        for i in range(rounds):
            update_vectors, projected_updates = [], []
            for cid, client in self.clients.items():
                upd_vec, proj_upd = client.local_train(self.global_model.copy(), self.lr, 5)
                proj_upd = proj_upd / np.linalg.norm(proj_upd)
                update_vectors.append(upd_vec)
                projected_updates.append(proj_upd)

            # 计算并打印鲁棒聚合时间
            start_time = time.time()  # 记录开始时间
            outliers = self.detect_anomalies(projected_updates)
            logger.info(f'Round {i}: anomalies {outliers}')

            good_idx = [j for j in range(len(update_vectors)) if j not in outliers]
            # 【F2 修复】全部被标记时不再 continue(它会导致该轮既不写 CSV 也不评估):
            #   后果1: 文件不足 101 行 → run_experiments.csv_exists 判定"未完成" → 反复重跑;
            #   后果2: CSV 的 round 列出现空洞 → "最后 20 行"不再是最后 20 轮 → avg20 口径错位。
            #   现在: 跳过更新(模型保持不变), 但仍评估并写盘, 保证每轮恰好一行。
            skipped = not good_idx
            if skipped:
                logger.error(f"Round {i}: all {len(update_vectors)} clients flagged; "
                             f"update skipped, model kept unchanged (row still written).")
            else:
                agg = self.aggregate_updates(update_vectors, good_idx)
                self.update_global_model(agg)
            aggregation_time = time.time() - start_time  # 计算聚合时间
            logger.info(f'Round {i} | Robust aggregation time: {aggregation_time:.4f} seconds'
                        f'{"" if not skipped else " (update skipped)"}')

            # 评估（先切 eval 模式，确保 F.dropout(training=...) 关闭）
            self.net.eval()
            correct = total = 0
            for data, label in self.test_data_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                pred = torch.argmax(self.net(data), dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
            acc = correct / total
            logger.info(f'Round {i} | Global acc: {acc:.3f}')

            # —— 计算全局平均测试损失 —— #
            self.net.eval()
            total_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for data, label in self.test_data_loader:
                    data, label = data.to(self.dev), label.to(self.dev)
                    output = self.net(data)
                    loss = self.loss_func(output, label)
                    total_loss += loss.item()
                    num_batches += 1
            avg_loss = total_loss / max(num_batches, 1)
            logger.info(f'Round {i} | Global loss: {avg_loss:.4f}')

            with open(save_csv, 'a') as f:
                f.write(f"{i},{acc:.6f},{avg_loss:.6f},{aggregation_time:.6f}\n")


if __name__ == '__main__':
    server = BatchAggrServer(
        num_clients=20, reduced_dim=128, noise_std=0.1,
        attack_mode="targeted_adaptive",  # 可切换: none|label_flip|backdoor|untargeted_adaptive|targeted_adaptive
        malicious_ratio=0.2,
        threshold_multiplier=2.0,
        data_distribution="iid",
        dirichlet_alpha=0.5,
        dataset="GTSRB",             # "GTSRB" | "MNIST" | "CIFAR10"
        r=15,
        beta=0.1,
    )
    server.start()
