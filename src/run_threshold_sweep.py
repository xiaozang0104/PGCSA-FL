"""
c（门限系数）横扫脚本
========================
论文门限公式: τ = c·s_med − s_min
扫 c = 1, 1.5, 2, 2.5, 3
×  3 个随机种子 (42, 123, 2026)
×  GTSRB + non_iid + targeted_adaptive
=  15 组实验

用法：改下面 C_VALUES / SEEDS，点运行。

CSV 命名: results/threshold_sweep_{DATASET}_c{c}_seed{seed}_{DIST}.csv
"""

import os
import sys

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  配置区                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
DATASET   = "GTSRB"
C_VALUES  = [1, 1.5, 2, 2.5, 3]
SEEDS     = [42, 123, 2026]
DIST      = "non_iid"
ATTACK    = "targeted_adaptive"
MAL_RATIO = 0.2

from server_test import BatchAggrServer


def _is_complete(path):
    """完整结果 = 表头 + 100 轮 = 101 行；半截 CSV（崩溃残留）视为未完成。"""
    try:
        with open(path) as f:
            return sum(1 for _ in f) >= 101
    except Exception:
        return False


def csv_name(c, seed):
    return f"results/threshold_sweep_{DATASET}_c{c}_seed{seed}_{DIST}.csv"


def run_one(c, seed):
    tag = f"c={c} | seed={seed} | {DATASET} | {ATTACK} | {DIST}"
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

    print(f"\n{'='*55}\n  threshold_sweep — ALL DONE ({total} configs)\n{'='*55}\n")


if __name__ == "__main__":
    main()
