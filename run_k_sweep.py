"""
k（reduced_dim，JL 投影维度）横扫脚本
========================================
扫 k = 64, 128, 256            [2026-09-14 更新]
    (不再跑 16/32: r=15 时 k=16 的 top-15 子空间几乎张满 16 维空间 → 残差≈0 判据退化;
     k=32 的 r/k=0.47 属边缘档, 待最终 r 确定后再决定是否补跑)
×  3 个随机种子 (42, 123, 2026)
×  GTSRB + non_iid + targeted_adaptive + mal 0.2
=  9 组实验
固定: r=15, beta=0.5, c=2.0

用法：改下面 K_VALUES / SEEDS，点运行。

CSV 命名: results/k_sweep_{DATASET}_k{k}_seed{seed}_{DIST}.csv
"""

import os
import sys

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  配置区                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
DATASET  = "GTSRB"
K_VALUES = [512]
SEEDS    = [2026]
DIST     = "non_iid"
ATTACK   = "targeted_adaptive"
MAL_RATIO = 0.4

from server_test import BatchAggrServer


def _is_complete(path):
    """完整结果 = 表头 + 100 轮 = 101 行；半截 CSV（崩溃残留）视为未完成。"""
    try:
        with open(path) as f:
            return sum(1 for _ in f) >= 101
    except Exception:
        return False


def csv_name(k, seed):
    return (f"results/k_sweep_{DATASET}_k{k}_mal{MAL_RATIO}_seed{seed}_{DIST}.csv")


def run_one(k, seed):
    tag = f"k={k} | seed={seed} | {DATASET} | {ATTACK} | {DIST}"
    banner = "=" * 55
    print(f"\n{banner}\n  {tag}\n{banner}\n")

    try:
        server = BatchAggrServer(
            num_clients=20,
            reduced_dim=k,
            noise_std=0.1,
            attack_mode=ATTACK,
            malicious_ratio=MAL_RATIO,
            data_distribution=DIST,
            dirichlet_alpha=0.5,
            dataset=DATASET,
            r=15,
            beta=0.5,
            threshold_multiplier=2.0,
            seed=seed,
        )
        server.start(save_csv=csv_name(k, seed))
    except Exception as e:
        print(f"  *** FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    total = len(K_VALUES) * len(SEEDS)
    done = 0

    for k in K_VALUES:
        for seed in SEEDS:
            done += 1
            fname = csv_name(k, seed)
            if os.path.exists(fname) and _is_complete(fname):
                print(f"[{done}/{total}] SKIP (exists)  k={k}  seed={seed}")
                continue
            print(f"[{done}/{total}] RUN  k={k}  seed={seed}")
            run_one(k, seed)

    print(f"\n{'='*55}\n  k_sweep — ALL DONE ({total} configs)\n{'='*55}\n")


if __name__ == "__main__":
    main()
