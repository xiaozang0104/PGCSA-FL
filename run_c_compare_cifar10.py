"""
c 对比测试 — CIFAR-10 40% T-Adaptive（r=15）
================================================
目的：验证 c=1 vs c=2 在 CIFAR-10 40% T-Adaptive 下的表现。
GTSRB 40% T-Adaptive 显示 c 敏感（c=1 略优、波动更小），需确认这是否
普适，还是 GTSRB 特有。CIFAR-10 模型弱(clean~70%)、良性噪声大，
是与 GTSRB（恶意伪装驱动）互补的场景。

配置：CIFAR10 + non_iid + targeted_adaptive + mal 0.4 + r=15 + β=0.9 + k=64
c ∈ {1, 2} × 3 seeds (42, 123, 2026) = 6 组实验

输出：results/c_compare_cifar10_mal40/c_compare_CIFAR10_c{c}_seed{s}_non_iid.csv
用法：python run_c_compare_cifar10.py，断点续跑。
"""

import os

from server_test import BatchAggrServer

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  配置区                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
DATASET   = "CIFAR10"
C_VALUES  = [1, 2]
SEEDS     = [42, 123, 2026]
DIST      = "non_iid"
ATTACK    = "targeted_adaptive"
MAL_RATIO = 0.4
OUT_DIR   = "results/c_compare_cifar10_mal40"

os.makedirs(OUT_DIR, exist_ok=True)


def _is_complete(path):
    """完整结果 = 表头 + 100 轮 = 101 行；半截 CSV（崩溃残留）视为未完成。"""
    try:
        with open(path) as f:
            return sum(1 for _ in f) >= 101
    except Exception:
        return False


def csv_name(c, seed):
    return os.path.join(OUT_DIR,
                        f"c_compare_CIFAR10_c{c:g}_seed{seed}_{DIST}.csv")


def run_one(c, seed):
    tag = f"c={c} | seed={seed} | {DATASET} | {ATTACK} | mal{MAL_RATIO} | {DIST} | r=15"
    banner = "=" * 55
    print(f"\n{banner}\n  {tag}\n{banner}\n")

    try:
        server = BatchAggrServer(
            num_clients=20,
            reduced_dim=128,
            noise_std=0.1,
            attack_mode=ATTACK,
            malicious_ratio=MAL_RATIO,
            data_distribution=DIST,
            dirichlet_alpha=0.5,
            dataset=DATASET,
            r=15,
            beta=0.1,
            threshold_multiplier=c,
            seed=seed,
        )
        server.start(save_csv=csv_name(c, seed))
    except Exception as e:
        print(f"  *** FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    total = len(C_VALUES) * len(SEEDS)
    done = 0

    for c in C_VALUES:
        for seed in SEEDS:
            done += 1
            fname = csv_name(c, seed)
            if os.path.exists(fname) and _is_complete(fname):
                print(f"[{done}/{total}] SKIP (exists)  c={c}  seed={seed}")
                continue
            print(f"[{done}/{total}] RUN  c={c}  seed={seed}")
            run_one(c, seed)

    print(f"\n{'='*55}\n  c_compare CIFAR10 mal40 — ALL DONE ({total} configs)\n"
          f"  结果在 {OUT_DIR}/\n{'='*55}\n")


if __name__ == "__main__":
    main()
