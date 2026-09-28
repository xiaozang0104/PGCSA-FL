"""
β（指数平滑系数）横扫脚本
===========================
扫 β = 0.1, 0.3, 0.5, 0.7, 0.9
×  3 个随机种子 (42, 123, 2026)
×  GTSRB + non_iid + targeted_adaptive
=  15 组实验

用法：改下面 BETA_VALUES / SEEDS，点运行。

CSV 命名: results/beta_sweep_{DATASET}_beta{beta}_seed{seed}_{DIST}.csv
"""

import os
import sys

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  配置区                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
DATASET     = "GTSRB"
BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS       = [42, 123, 2026]
DIST        = "non_iid"
ATTACK      = "targeted_adaptive"
MAL_RATIO   = 0.4

from server_test import BatchAggrServer


def _is_complete(path):
    """完整结果 = 表头 + 100 轮 = 101 行；半截 CSV（崩溃残留）视为未完成。"""
    try:
        with open(path) as f:
            return sum(1 for _ in f) >= 101
    except Exception:
        return False


def csv_name(beta, seed):
    return (f"results/beta_sweep_{DATASET}_b{beta}_mal{MAL_RATIO}_seed{seed}_{DIST}.csv")


def run_one(beta, seed):
    tag = f"β={beta} | seed={seed} | {DATASET} | {ATTACK} | {DIST}"
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
            beta=beta,
            threshold_multiplier=2.0,
            seed=seed,
        )
        server.start(save_csv=csv_name(beta, seed))
    except Exception as e:
        print(f"  *** FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    total = len(BETA_VALUES) * len(SEEDS)
    done = 0

    for beta in BETA_VALUES:
        for seed in SEEDS:
            done += 1
            fname = csv_name(beta, seed)
            if os.path.exists(fname) and _is_complete(fname):
                print(f"[{done}/{total}] SKIP (exists)  β={beta}  seed={seed}")
                continue
            print(f"[{done}/{total}] RUN  β={beta}  seed={seed}")
            run_one(beta, seed)

    print(f"\n{'='*55}\n  beta_sweep — ALL DONE ({total} configs)\n{'='*55}\n")


if __name__ == "__main__":
    main()
