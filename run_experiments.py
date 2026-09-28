"""
一键批跑脚本 — 统一实验矩阵（18 方法 × 3 数据集 × 17 攻击/比例组合 × 2 分布 × 3 seed）
========================================================================
  METHOD = "krum"          →  只跑 Krum
  METHOD = "foundationfl"  →  只跑 FoundationFL
  METHOD = "qin"           →  只跑 Qin et al.
  METHOD = "pgcsa"         →  只跑 PGCSA-FL（你自己的方法）
  METHOD = "trimmed_mean"  →  只跑 Trimmed Mean
  METHOD = "median"        →  只跑 Median
  METHOD = "all"           →  按顺序跑全部

矩阵维度：
  方法 × 数据集(MNIST/GTSRB/CIFAR10) × 攻击(none/label_flip/backdoor/
  untargeted_adaptive/targeted_adaptive) × 恶意比例(0/0.1/0.2/0.3/0.4)
  × 分布(iid/non_iid) × seed(42/123/2026)
  配对规则：0% 只跑 none；>0% 跑 4 种攻击（17 组合/方法/数据集/分布）
  每个方法 = 3 × 17 × 2 × 3 = 306 次运行；全部 18 方法 = 5508 次。

CSV 命名: results/{method}_{dataset}_{attack}_mal{ratio}_{dist}_seed{seed}.csv

特性：自动跳过已有 CSV（断点续跑，粒度 = 单个 seed），崩了继续下一个。
按需取子集：改 ONLY_RATIOS / ONLY_ATTACKS 两个变量（见下方配置区），
例：ONLY_RATIOS=[0.4] + ONLY_ATTACKS=5 种攻击 → 只跑 40% 恶意 × 全攻击。
========================================================================
"""
import sys
import os
import importlib

METHOD = "pgcsa"      # krum | foundationfl | qin | pgcsa | trimmed_mean | median | dnc | sybil | fltrust | pefl | shieldfl | rfed | lsfl | split | fedavg_dp | dpdlda | pflf | esfl | all

# ---------- 实验矩阵 ----------
DATASETS   = ["MNIST"]
DISTS      = ["non_iid"]
SEEDS      = [2026]

# 恶意比例 × 攻击配对：0% 无恶意客户端 → 只跑 none；>0% → 4 种攻击
RATIO_ATTACKS = [
    (0.0, ["none"]),
    (0.1, ["label_flip", "backdoor", "untargeted_adaptive", "targeted_adaptive"]),
    (0.2, ["label_flip", "backdoor", "untargeted_adaptive", "targeted_adaptive"]),
    (0.3, ["label_flip", "backdoor", "untargeted_adaptive", "targeted_adaptive"]),
    (0.4, ["label_flip", "backdoor", "untargeted_adaptive", "targeted_adaptive"]),
]

# ---------- 运行范围过滤（按需取子集；None = 跑完整矩阵） ----------
# 例：只跑 40% 恶意比例 × 全部 5 种攻击（含 none 基线）：
#   ONLY_RATIOS  = [0.4]
#   ONLY_ATTACKS = ["none", "label_flip", "backdoor",
#                   "untargeted_adaptive", "targeted_adaptive"]
ONLY_RATIOS  = [0.4]   # PFLF：CIFAR-10 non_iid 全比例（17 组合 × 3 seed = 51）
ONLY_ATTACKS = ["targeted_adaptive"]

# ---------- 方法注册 ----------
# 每个方法:  (module, class_name, extra_kwargs)
METHOD_REGISTRY = {
    "krum": {
        "module": "run_krum",
        "class": "KrumServer",
        "kwargs": {},
        "use_start": False,
    },
    "foundationfl": {
        "module": "run_foundationfl",
        "class": "FoundationFLServer",
        "kwargs": {"ff_rule": "trimmed_mean"},
        "use_start": False,
    },
    "qin": {
        "module": "run_qin",
        "class": "QinServer",
        "kwargs": {"trusted_size": 100, "use_jl": True, "reduced_dim": 64},
        "use_start": False,
    },
    "pgcsa": {
        "module": "server_test",
        "class": "BatchAggrServer",
        "kwargs": {"reduced_dim": 256, "noise_std": 0.1, "beta": 0.1,
                   "threshold_multiplier": 2.0, "r": 2},
        "use_start": True,   # PGCSA-FL uses .start() instead of .run()
    },
    "trimmed_mean": {
        "module": "run_trimmed_mean",
        "class": "TrimmedMeanServer",
        "kwargs": {"agg_rule": "trimmed_mean"},
        "use_start": False,
    },
    "median": {
        "module": "run_trimmed_mean",
        "class": "TrimmedMeanServer",
        "kwargs": {"agg_rule": "median"},
        "use_start": False,
    },
    "dnc": {
        "module": "run_dnc",
        "class": "DncServer",
        "kwargs": {"n_iters": 1, "c": 1, "b": 10000},
        "use_start": False,
    },
    "sybil": {
        "module": "run_sybil",
        "class": "SybilServer",
        "kwargs": {},
        "use_start": False,
    },
    "fltrust": {
        "module": "run_fltrust",
        "class": "FLTrustServer",
        "kwargs": {"root_size": 100},
        "use_start": False,
    },
    "pefl": {
        "module": "run_pefl",
        "class": "PEFLServer",
        "kwargs": {},
        "use_start": False,
    },
    "shieldfl": {
        "module": "run_shieldfl",
        "class": "ShieldFLServer",
        "kwargs": {},
        "use_start": False,
    },
    "rfed": {
        "module": "run_rfed",
        "class": "RFedServer",
        "kwargs": {"root_size": 100},
        "use_start": False,
    },
    "lsfl": {
        "module": "run_lsfl",
        "class": "LSFLServer",
        "kwargs": {"k_frac": 0.7},
        "use_start": False,
    },
    "split": {
        "module": "run_split",
        "class": "SplitServer",
        "kwargs": {},
        "use_start": False,
    },
    "fedavg_dp": {
        "module": "run_fedavg_dp",
        "class": "FedAvgDPServer",
        "kwargs": {"clip_norm": 1.0, "noise_std": 0.1},
        "use_start": False,
    },
    "dpdlda": {
        "module": "run_dpdlda",
        "class": "DPDLDAServer",
        "kwargs": {"clip_norm": 4.0, "n_groups": 8, "eta": 0.5, "sigma": 0.1},
        "use_start": False,
    },
    "pflf": {
        "module": "run_pflf",
        "class": "PFLFServer",
        "kwargs": {"mu": 0.01, "noise_std": 0.1},
        "use_start": False,
    },
    "esfl": {
        "module": "run_esfl",
        "class": "ESFLServer",
        "kwargs": {"sigma": 0.1, "epsilon": 2.0},
        "use_start": False,
    },
}


def csv_exists(method, dataset, attack, ratio, dist, seed):
    """Check whether this experiment's CSV already exists AND is complete.

    只检查文件存在会误判：训练中途崩溃会留下 <100 轮的半截 CSV，
    若不检查完整性会被永久跳过。完整文件 = 表头 + 100 轮 = 101 行。
    """
    path = (f"results/{method}_{dataset}_{attack}_"
            f"mal{ratio}_{dist}_seed{seed}.csv")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            n_lines = sum(1 for _ in f)
    except Exception:
        return False
    return n_lines >= 101  # header + 100 rounds


def run_one(method, dataset, attack, ratio, dist, seed,
            server_cls, extra_kwargs, use_start):
    """Run a single experiment configuration (method/dataset/attack/ratio/dist/seed)."""
    tag = f"{method} | {dataset} | {attack} | mal{ratio} | {dist} | seed{seed}"
    banner = "=" * 65
    print(f"\n{banner}\n  {tag}\n{banner}\n")

    try:
        kwargs = dict(
            num_clients=20,
            attack_mode=attack,
            malicious_ratio=ratio,
            data_distribution=dist,
            dirichlet_alpha=0.5,
            dataset=dataset,
            seed=seed,
            **extra_kwargs,
        )
        if method == "qin":
            kwargs["jl_seed"] = seed  # JL 矩阵随主 seed 变化，与 PGCSA 的 JL 行为一致

        server = server_cls(**kwargs)

        save_csv = (f"results/{method}_{dataset}_{attack}_"
                    f"mal{ratio}_{dist}_seed{seed}.csv")
        if use_start:
            server.start(save_csv=save_csv)
        else:
            server.run(model_class=None, rounds=100, local_epochs=5,
                       method_name=method, save_csv=save_csv)
    except Exception as e:
        print(f"  ***  FAILED: {e}")
        import traceback
        traceback.print_exc()


def run_method(method):
    """Run all configs for one method, respecting ONLY_RATIOS/ONLY_ATTACKS filters."""
    info = METHOD_REGISTRY[method]
    mod = importlib.import_module(info["module"])
    server_cls = getattr(mod, info["class"])
    extra_kwargs = info["kwargs"]
    use_start = info["use_start"]

    # 应用过滤，得到有效的 (ratio, attacks) 组合列表
    combos = []
    for ratio, attacks in RATIO_ATTACKS:
        if ONLY_RATIOS is not None and ratio not in ONLY_RATIOS:
            continue
        if ONLY_ATTACKS is not None:
            attacks = ONLY_ATTACKS
        combos.append((ratio, attacks))

    total = len(DATASETS) * sum(len(a) for _, a in combos) * len(DISTS) * len(SEEDS)
    done = 0

    for dataset in DATASETS:
        for ratio, attacks in combos:
            for attack in attacks:
                for dist in DISTS:
                    for seed in SEEDS:
                        done += 1
                        if csv_exists(method, dataset, attack, ratio, dist, seed):
                            print(f"[{done}/{total}] SKIP (exists)  {method} | {dataset} | {attack} | mal{ratio} | {dist} | seed{seed}")
                            continue
                        print(f"[{done}/{total}] RUN  {method} | {dataset} | {attack} | mal{ratio} | {dist} | seed{seed}")
                        run_one(method, dataset, attack, ratio, dist, seed,
                                server_cls, extra_kwargs, use_start)

    print(f"\n{'='*65}\n  {method} — ALL DONE ({total} configs)\n{'='*65}\n")


# =====================================================================
if __name__ == "__main__":
    # 终端传参优先，没传参就用脚本里设置的 METHOD
    target = sys.argv[1].lower() if len(sys.argv) > 1 else METHOD
    target = target.strip()

    if target == "all":
        for m in METHOD_REGISTRY:
            run_method(m)
    elif target in METHOD_REGISTRY:
        run_method(target)
    else:
        print(f"未知方法 '{target}'，可用: {list(METHOD_REGISTRY.keys())} | all")
        sys.exit(1)
