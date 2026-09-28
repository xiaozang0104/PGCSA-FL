"""
r 值横扫脚本（有攻击版，多 seed）
==================================
扫 r = 1, 5, 10, 15, 20, 25, 30
×  3 个随机种子 (42, 123, 2026)
×  GTSRB + non_iid + targeted_adaptive
=  21 组实验

用法：改下面 DATASET 和 R_VALUES / SEEDS，点运行。

CSV 命名: results/r_sweep_attack_{DATASET}_r{r}_seed{seed}_{DIST}.csv
"""

import os
import sys

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  改这里                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
DATASET   = "GTSRB"    # MNIST | GTSRB | CIFAR10
R_VALUES  = [3]
SEEDS     = [2026]
DIST      = "non_iid"
ATTACK    = "targeted_adaptive"
MAL_RATIO = 0.4

from server_test import BatchAggrServer


def _is_complete(path):
    """完整结果 = 表头 + 100 轮 = 101 行；半截 CSV（崩溃残留）视为未完成。"""
    try:
        with open(path) as f:
            return sum(1 for _ in f) >= 101
    except Exception:
        return False


def csv_name(r, seed):
    return f"results/r_sweep_attack_{DATASET}_r{r}_mal{MAL_RATIO}_seed{seed}_{DIST}.csv"


def run_one(r, seed):
    tag = f"r={r:2d} | seed={seed} | {DATASET} | {ATTACK} | {DIST}"
    banner = "=" * 55
    print(f"\n{banner}\n  {tag}\n{banner}\n")

    try:
        server = BatchAggrServer(
            num_clients=20,
            reduced_dim=64,
            noise_std=0.1,
            attack_mode=ATTACK,
            malicious_ratio=MAL_RATIO,
            data_distribution=DIST,
            dirichlet_alpha=0.5,
            dataset=DATASET,
            r=r,
            beta=0.9,
            threshold_multiplier=2.0,
            seed=seed,
        )
        server.start(save_csv=csv_name(r, seed))
    except Exception as e:
        print(f"  *** FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    total = len(R_VALUES) * len(SEEDS)
    done = 0

    for r in R_VALUES:
        for seed in SEEDS:
            done += 1
            fname = csv_name(r, seed)
            if os.path.exists(fname) and _is_complete(fname):
                print(f"[{done}/{total}] SKIP (exists)  r={r:2d}  seed={seed}")
                continue
            print(f"[{done}/{total}] RUN  r={r:2d}  seed={seed}")
            run_one(r, seed)

    print(f"\n{'='*55}\n  r_sweep_attack — ALL DONE ({total} configs)\n{'='*55}\n")


if __name__ == "__main__":
    main()
