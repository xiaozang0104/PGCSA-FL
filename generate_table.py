"""
从 results/ 的 CSV 生成统一实验表格（3-seed 均值±标准差）
========================================================
读取所有 {method}_{dataset}_{attack}_mal{ratio}_{dist}_seed{seed}.csv，
对每个 (方法, 数据集, 分布, 比例, 攻击) 配置：
  1) 每个 seed 文件算最后 20 轮 (round 80-99) 的准确率均值 (avg20)
  2) 聚合 3 个 seed → 均值 ± 标准差 (ddof=1)

输出 Excel（两个 sheet）：
  - main_mal0.2 : 恶意比例 20% 主表（3 数据集 × 2 分布 × 4 攻击，18 方法列）
  - master      : 全矩阵（3 数据集 × 2 分布 × 17 比例/攻击组合，18 方法列）

用法: python generate_table.py
输出: Table_all_avg20.xlsx
"""

import os
import glob
import numpy as np
import openpyxl
from openpyxl.styles import Font, Alignment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Table_all_avg20.xlsx")

SEEDS = [42, 123, 2026]

# ---- 方法映射（文件名 → 表格列名），顺序即表格列顺序 ----
METHOD_MAP = [
    ("krum",          "Krum"),
    ("trimmed_mean",  "Trimmed-means"),
    ("median",        "Median"),
    ("dnc",           "Dnc"),
    ("sybil",         "Sybil"),
    ("fltrust",       "FLTrust"),
    ("foundationfl",  "FoundationFL"),
    ("pefl",          "PEFL"),
    ("shieldfl",      "ShieldFL"),
    ("rfed",          "RFed"),
    ("lsfl",           "LSFL"),
    ("split",         "Split"),
    ("qin",           "Qin"),
    ("pgcsa",         "Ours"),
    ("fedavg_dp",     "FedAvg+DP"),
    ("dpdlda",        "DPDLDA"),
    ("pflf",          "PFLF"),
    ("esfl",          "ESFL"),
]

METHOD_NAMES = [m[0] for m in METHOD_MAP]
DATASETS = ["MNIST", "GTSRB", "CIFAR10"]
DISTS    = ["iid", "non_iid"]
ATTACKS  = ["none", "label_flip", "backdoor",
            "untargeted_adaptive", "targeted_adaptive"]
RATIOS   = [0.0, 0.1, 0.2, 0.3, 0.4]

# 主表 (mal0.2) 只展示 >0% 的 4 种攻击（none 只在 0% 档跑）
MAIN_ATTACKS = ["label_flip", "backdoor",
                "untargeted_adaptive", "targeted_adaptive"]

# 每方法应出现的 CSV 数量：3 数据集 × 17 组合 × 2 分布 × 3 seed = 306
EXPECTED_PER_METHOD = (len(DATASETS) *
                       sum(1 if r == 0.0 else len(ATTACKS) - 1 for r in RATIOS) *
                       len(DISTS) * len(SEEDS))


def parse_filename(base):
    """解析 results/{method}_{dataset}_{attack}_mal{ratio}_{dist}_seed{seed}.csv

    返回 dict(method, dataset, attack, ratio, dist, seed)；无法解析返回 None。
    """
    if not base.endswith(".csv"):
        return None
    rest = base[:-len(".csv")]

    # 方法前缀（先匹配长的，避免 trimmed_mean 等带下划线名被误拆）
    method = None
    for m in sorted(METHOD_NAMES, key=len, reverse=True):
        if rest.startswith(m + "_"):
            method = m
            rest = rest[len(m) + 1:]
            break
    if method is None:
        return None

    # rest = {dataset}_{attack}_mal{ratio}_{dist}_seed{seed}
    if "_mal" not in rest:
        return None
    head, tail = rest.split("_mal", 1)

    dataset = None
    for ds in DATASETS:
        if head.startswith(ds + "_"):
            dataset = ds
            attack = head[len(ds) + 1:]
            break
    if dataset is None:
        return None

    # tail = {ratio}_{dist}_seed{seed}
    if "_seed" not in tail:
        return None
    head2, seed_str = tail.rsplit("_seed", 1)
    try:
        seed = int(seed_str)
    except ValueError:
        return None

    if head2.endswith("non_iid"):
        dist = "non_iid"
        ratio = float(head2[:-len("non_iid") - 1])
    elif head2.endswith("_iid"):
        dist = "iid"
        ratio = float(head2[:-len("_iid")])
    else:
        return None

    return dict(method=method, dataset=dataset, attack=attack,
                ratio=ratio, dist=dist, seed=seed)


def avg20_of_csv(path):
    """读取 CSV 的 accuracy 列，返回最后 20 轮均值。

    只接受完整的 100 轮结果（101 行含表头）；半截 CSV（训练中途
    崩溃）直接视为无效返回 None，避免把中间轮次误当 avg20。
    """
    accs = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    accs.append(float(parts[1]))
                except ValueError:
                    continue
    if len(accs) < 100:
        return None
    return float(np.mean(accs[-20:]))


def mean_std_str(avg20s):
    """avg20s: list[float]（跨 seed）→ 'm ± s' 字符串。"""
    if not avg20s:
        return ""
    a = np.asarray(avg20s, dtype=float)
    m = a.mean()
    s = float(a.std(ddof=1)) if len(a) >= 2 else 0.0
    return f"{m:.2f} ± {s:.2f}"


def scan_results():
    """扫描 results/，返回 data[(method,dataset,dist,ratio,attack)] = {seed: avg20}
    以及每个方法出现的文件计数。"""
    data = {}
    counts = {m: 0 for m in METHOD_NAMES}
    seed_counts = {}  # (method,dataset,dist,ratio,attack) -> list of seeds seen

    for f in glob.glob(os.path.join(RESULTS_DIR, "*.csv")):
        info = parse_filename(os.path.basename(f))
        if info is None:
            continue
        avg20 = avg20_of_csv(f)
        if avg20 is None:
            continue
        key = (info["method"], info["dataset"], info["dist"],
               info["ratio"], info["attack"])
        data.setdefault(key, {})[info["seed"]] = avg20
        seed_counts.setdefault(key, []).append(info["seed"])
        counts[info["method"]] += 1
    return data, counts, seed_counts


def build_main_sheet(ws, data):
    """main_mal0.2：rows = 数据集×分布×攻击(4)，cols = 18 方法。"""
    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws.cell(1, 1, "Dataset")
    ws.cell(1, 2, "Attack")
    ws.cell(1, 3, "Robust FL")
    ws.merge_cells(start_row=1, start_column=3, end_row=1, end_column=9)
    ws.cell(1, 10, "Robust and Privacy-Preserving FL")
    ws.merge_cells(start_row=1, start_column=10, end_row=1, end_column=20)

    for j, (_, col_name) in enumerate(METHOD_MAP):
        ws.cell(2, j + 3, col_name)

    for r in (1, 2):
        for c in range(1, 21):
            cell = ws.cell(r, c)
            cell.font = bold
            cell.alignment = center

    row = 3
    for dataset in DATASETS:
        for dist in DISTS:
            for attack in MAIN_ATTACKS:
                ws.cell(row, 1, f"{dataset}\n{dist}")
                ws.cell(row, 2, attack)
                for j, (method, _) in enumerate(METHOD_MAP):
                    key = (method, dataset, dist, 0.2, attack)
                    ws.cell(row, j + 3, mean_std_str(list(data.get(key, {}).values())))
                row += 1

    for r in range(3, row):
        for c in range(1, 21):
            cell = ws.cell(r, c)
            cell.alignment = (center if c == 1 else
                              Alignment(horizontal="left", vertical="center")
                              if c == 2 else center)

    ws.column_dimensions['A'].width = 14
    ws.column_dimensions['B'].width = 14
    for col in "CDEFGHIJKLMNOPQRST":
        ws.column_dimensions[col].width = 15


def build_master_sheet(ws, data):
    """master：rows = 数据集×分布×比例×攻击（102 行），cols = 18 方法。"""
    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    headers = ["Dataset", "Dist", "Ratio", "Attack"] + [c for _, c in METHOD_MAP]
    for c, h in enumerate(headers, 1):
        cell = ws.cell(1, c, h)
        cell.font = bold
        cell.alignment = center

    row = 2
    for dataset in DATASETS:
        for dist in DISTS:
            for ratio in RATIOS:
                # 展示全部 5 种攻击；未跑的组合（如 0% 下的 4 种攻击）留空
                for attack in ATTACKS:
                    ws.cell(row, 1, dataset)
                    ws.cell(row, 2, dist)
                    ws.cell(row, 3, ratio)
                    ws.cell(row, 4, attack)
                    for j, (method, _) in enumerate(METHOD_MAP):
                        key = (method, dataset, dist, ratio, attack)
                        ws.cell(row, j + 5, mean_std_str(list(data.get(key, {}).values())))
                    row += 1

    for r in range(2, row):
        for c in range(1, 5 + len(METHOD_MAP)):
            ws.cell(r, c).alignment = center
    for c in range(1, 5 + len(METHOD_MAP)):
        ws.column_dimensions[openpyxl.utils.get_column_letter(c)].width = 15


def main():
    data, counts, seed_counts = scan_results()

    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "main_mal0.2"
    build_main_sheet(ws1, data)

    ws2 = wb.create_sheet("master")
    build_master_sheet(ws2, data)

    wb.save(OUTPUT)
    print(f"→ 已生成 {OUTPUT}")

    print("\n各方法覆盖情况（每个方法应 306 个 CSV）：")
    for method, col in METHOD_MAP:
        print(f"  {col:15s} ({method:15s}): {counts.get(method, 0)}/{EXPECTED_PER_METHOD}")

    print("\n配置级 seed 覆盖（应有 3 个 seed，不足的列出）：")
    incomplete = {k: sorted(v) for k, v in seed_counts.items() if len(set(v)) < len(SEEDS)}
    for k, seeds in sorted(incomplete.items()):
        print(f"  {k}: seeds={seeds}")


if __name__ == "__main__":
    main()
