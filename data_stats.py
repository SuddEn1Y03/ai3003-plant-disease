#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_stats.py
=============
统计 PlantVillage / PlantDoc 中番茄三类（healthy / early_blight / late_blight）
的图片数量，输出 CSV 与 Markdown 表格。

设计原则
--------
1. 不猜测目录结构：脚本会自己扫描目录，把"直接包含图片的文件夹"当成一个类别目录，
   并把它的父目录名（train / test / color ...）当成 split 名。
2. 类名映射写在 DEFAULT_MAP 里只是"草稿"，真实类名必须人工核对后再运行；
   脚本会把**没有匹配上的原始类名**打印出来，方便你发现 PlantDoc 的命名差异。
3. 只统计，不修改、不移动任何图片；对损坏图片只计数不报错。

用法
----
    # 1) 先看数据集里到底有哪些类名（不统计，只列目录）
    python data_stats.py --root plantvillage=<path>/PlantVillage-Dataset/raw/color \
                         --root plantdoc=<path>/PlantDoc-Dataset --list-only

    # 2) 正式统计，输出 CSV + Markdown
    python data_stats.py --root plantvillage=<path>/PlantVillage-Dataset/raw/color \
                         --root plantdoc=<path>/PlantDoc-Dataset \
                         --out-csv stats/class_counts.csv \
                         --out-md  stats/class_counts.md

    # 3) 使用自己的映射文件（JSON，格式见 --dump-mapping-template）
    python data_stats.py ... --mapping my_mapping.json

输出
----
* stats/class_counts.csv   机器可读，方便贴到 Excel 或画图
* stats/class_counts.md    可以直接复制进报告
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

UNIFIED_LABELS = ["healthy", "early_blight", "late_blight"]

# 统一标签的中文/英文显示名，方便直接写报告
LABEL_DISPLAY = {
    "healthy": "healthy (健康)",
    "early_blight": "early_blight (早疫病)",
    "late_blight": "late_blight (晚疫病)",
}

# split / 变体名：出现在路径里就作为一列
SPLIT_NAMES = {"train", "test", "val", "valid", "validation", "training", "testing"}
VARIANT_NAMES = {"color", "grayscale", "segmented"}
# 默认不统计非彩色变体（PlantVillage 仓库自带 color / grayscale / segmented 三份拷贝，
# 全算进去会让样本数变成 3 倍，属于典型的数据统计错误）
SKIP_VARIANTS_BY_DEFAULT = {"grayscale", "segmented"}

# --------------------------------------------------------------------------- #
# 类名映射草稿（⚠ 必须人工核对真实目录名！）
# --------------------------------------------------------------------------- #
# PlantVillage 原始类名格式为 "<Crop>___<Disease>"，番茄三类如下
#   Tomato___healthy / Tomato___Early_blight / Tomato___Late_blight
# PlantDoc 的目录名是自然语言短语，不同 fork / 版本之间**不一致**，
# 下面只是常见写法的草稿，运行 --list-only 后请按实际输出改这里或用 --mapping。
DEFAULT_MAP = {
    "plantvillage": {
        "Tomato___healthy": "healthy",
        "Tomato___Early_blight": "early_blight",
        "Tomato___Late_blight": "late_blight",
        # 有些仓库版本会把类别名写成带空格的形式，作为兜底：
        "Tomato_healthy": "healthy",
        "Tomato_Early_blight": "early_blight",
        "Tomato_Late_blight": "late_blight",
        "Tomato healthy": "healthy",
        "Tomato Early blight": "early_blight",
        "Tomato Late blight": "late_blight",
    },
    "plantdoc": {
        # 已核对（读 GitHub 仓库 master 的 train/ 目录列表，2026-11）：
        # PlantDoc 的 train/ 下共有 9 个番茄目录，下面这 3 个是本项目要用的。
        # 注意：PlantDoc 里没有 "healthy" 字样，健康叶就叫 "Tomato leaf"。
        "Tomato leaf": "healthy",
        "Tomato Early blight leaf": "early_blight",
        "Tomato leaf late blight": "late_blight",
        # 以下为不同 fork / 旧镜像里可能出现的写法，命中也不影响结果：
        "Tomato healthy": "healthy",
        "Tomato leaf healthy": "healthy",
        "Tomato Early blight": "early_blight",
        "Tomato Late blight leaf": "late_blight",
        "Tomato late blight leaf": "late_blight",
    },
}


# --------------------------------------------------------------------------- #
# 目录扫描
# --------------------------------------------------------------------------- #
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def count_images(d: Path) -> int:
    n = 0
    for child in d.iterdir():
        if child.is_file() and is_image(child):
            n += 1
    return n


def scan_class_dirs(root: Path, skip_variants: set[str]):
    """
    扫描 root 下所有"直接包含图片"的目录，返回 [(split, class_name, dir, n_images)]。

    * split   : 路径里出现的 train / test / val / color 之一，找不到就记 "all"
    * class   : 该目录名（即原始类名）
    * 若目录名属于 grayscale / segmented 且被跳过，会记在 skipped 里返回

    特例：数据集根目录自己也可能直接放着一两张图（例如 PlantDoc 仓库根目录的
    `PlantDoc_Examples.png` 示例图）。只要根目录下面还有真正的类目录，就不把根目录
    当成一个类别，否则它会在"未匹配"告警里冒充一个类，干扰判断。
    反过来，如果用户直接把某个类目录当 --root 传进来（下面没有任何含图的子目录），
    那这个根目录本身就是要统计的类。
    """
    found, skipped = [], []
    root = root.resolve()
    root_hit = None
    for cur_dir, sub_dirs, files in os.walk(root):
        # 跳过隐藏目录（.git / .ipynb_checkpoints 等）
        sub_dirs[:] = [s for s in sub_dirs if not s.startswith(".")]
        cur = Path(cur_dir)
        if not any(is_image(cur / f) for f in files if (cur / f).is_file()):
            continue
        rel_parts = cur.relative_to(root).parts
        split = "all"
        for part in rel_parts:
            low = part.lower()
            if low in SPLIT_NAMES:
                split = low
                break
            if low in VARIANT_NAMES:
                split = low
                break
        if any(p.lower() in skip_variants for p in rel_parts):
            skipped.append((str(cur), count_images(cur)))
            continue
        if cur == root:
            root_hit = (split, cur.name, cur, count_images(cur))
            continue
        found.append((split, cur.name, cur, count_images(cur)))
    if not found and root_hit is not None:
        found.append(root_hit)      # 用户直接指定的就是类目录
    return found, skipped


# --------------------------------------------------------------------------- #
# 映射
# --------------------------------------------------------------------------- #
def load_mapping(path: Path | None) -> dict:
    """读取自定义映射 JSON；没有就用 DEFAULT_MAP。"""
    if path is None:
        return {k: dict(v) for k, v in DEFAULT_MAP.items()}
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    merged = {k: dict(v) for k, v in DEFAULT_MAP.items()}
    for ds, m in user.items():
        merged.setdefault(ds, {}).update(m)
    return merged


def normalize_class_name(name: str) -> str:
    """用于'模糊比对'的归一化：小写、去空格/下划线/连字符、折叠多空格。"""
    s = name.strip().lower().replace("___", " ").replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def build_lookup(ds_mapping: dict) -> dict:
    return {normalize_class_name(k): v for k, v in ds_mapping.items()}


# --------------------------------------------------------------------------- #
# 统计主流程
# --------------------------------------------------------------------------- #
def collect(roots: list[tuple[str, Path]], mapping: dict, skip_variants: set[str],
            list_only: bool = False):
    rows = []            # 每条 = 一个 (dataset, split, raw_class)
    unmatched = []       # 未命中映射的 (dataset, raw_class, n)
    skipped_all = []

    for ds_name, root in roots:
        if not root.exists():
            print(f"[WARN] 数据集路径不存在，已跳过：{ds_name} -> {root}", file=sys.stderr)
            continue
        ds_map = mapping.get(ds_name, {})
        lookup = build_lookup(ds_map)
        found, skipped = scan_class_dirs(root, skip_variants)
        skipped_all.extend([(ds_name, *s) for s in skipped])
        if not found:
            print(f"[WARN] 在 {root} 下没找到任何含图片的目录，"
                  f"请检查是否解压到了更深一层（例如 */PlantVillage-Dataset/raw/color）。",
                  file=sys.stderr)
        # 同一 (split, class) 可能出现多个目录，合并计数
        agg: "OrderedDict[tuple[str, str], int]" = OrderedDict()
        for split, cls, _d, n in sorted(found, key=lambda x: (x[0], x[1])):
            agg[(split, cls)] = agg.get((split, cls), 0) + n
        for (split, cls), n in agg.items():
            unified = lookup.get(normalize_class_name(cls))
            if unified is None:
                unmatched.append((ds_name, split, cls, n))
            rows.append(dict(dataset=ds_name, split=split, raw_class=cls,
                             unified_label=unified or "", n_images=n))
        if list_only:
            print(f"\n=== {ds_name} : {root} ===")
            for (split, cls), n in agg.items():
                hit = lookup.get(normalize_class_name(cls), "-- 未匹配 --")
                print(f"  [{split:9s}] {cls:38s} n={n:6d}  ->  {hit}")
    return rows, unmatched, skipped_all


def summarize(rows: list[dict]) -> list[dict]:
    """按 (dataset, unified_label) 汇总，只统计映射成功的行。"""
    agg: "OrderedDict[tuple[str, str], int]" = OrderedDict()
    for r in rows:
        if not r["unified_label"]:
            continue
        key = (r["dataset"], r["unified_label"])
        agg[key] = agg.get(key, 0) + r["n_images"]
    out = []
    for (ds, lab), n in agg.items():
        out.append(dict(dataset=ds, unified_label=lab, n_images=n))
    return out


def write_csv(rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "split", "raw_class", "unified_label", "n_images"]
    # utf-8-sig 让 Excel 打开不乱码
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[OK] CSV 已写入：{out_path}")


def markdown_table(header: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in body:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_markdown(rows: list[dict], out_path: Path,
                   unmatched: list[tuple], skipped: list[tuple],
                   note: str = ""):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# 数据集样本统计（自动生成）\n")
    md.append(f"- 生成脚本：`data_stats.py`\n- 统计口径：仅统计图片文件，"
              f"扩展名 {sorted(IMAGE_EXTS)}\n")
    if note:
        md.append(f"- 数据根目录：{note}\n")

    datasets = list(OrderedDict.fromkeys(r["dataset"] for r in rows))
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] == ds]
        total = sum(r["n_images"] for r in sub)
        md.append(f"\n\n## {ds}（合计 {total} 张）\n")
        body = [[r["split"], r["raw_class"], r["unified_label"] or "**[未匹配]**",
                 r["n_images"]] for r in sub]
        md.append(markdown_table(["split", "原始类名", "统一标签", "图片数"], body))
        md.append(f"\n\n> 小计：**{total}** 张；其中已映射 "
                  f"**{sum(r['n_images'] for r in sub if r['unified_label'])}** 张。\n")

    md.append("\n\n## 统一标签汇总（用于报告 Table 1）\n")
    summ = summarize(rows)
    table = {}
    for s in summ:
        table[(s["dataset"], s["unified_label"])] = s["n_images"]
    body, totals = [], [0] * len(datasets)
    for lab in UNIFIED_LABELS:
        line = [LABEL_DISPLAY[lab]]
        for j, ds in enumerate(datasets):
            n = table.get((ds, lab), 0)
            line.append(n)
            totals[j] += n
        body.append(line)
    body.append(["合计"] + totals)      # 每列一个总数，列数必须与表头一致
    md.append(markdown_table(["统一标签"] + datasets, body))

    if unmatched:
        md.append("\n\n## ⚠ 未匹配到统一标签的目录（请人工核对）\n")
        md.append(markdown_table(["数据集", "split", "原始类名", "图片数"],
                                 [[u[0], u[1], u[2], u[3]] for u in unmatched]))
    if skipped:
        md.append("\n\n## 已跳过的非彩色变体（默认不计入）\n")
        md.append(markdown_table(["数据集", "目录", "图片数"],
                                 [[s[0], s[1], s[2]] for s in skipped]))
    md.append("\n\n## 待填项\n")
    md.append("- [待填：核对以上数字后，把本表复制进报告 3.2 节 Table 1]\n")
    md.append("- [待填：说明是否删除了损坏/非叶片图片，删除了几张]\n")
    md.append("- [待填：说明 PlantVillage 重复/近重复图片的检查方法与结果]\n")

    out_path.write_text("".join(md), encoding="utf-8")
    print(f"[OK] Markdown 已写入：{out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_root_arg(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            "格式应为 NAME=PATH，例如 plantvillage=D:/data/PlantVillage/color")
    name, path = s.split("=", 1)
    return name.strip().lower(), Path(path.strip().strip('"'))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="统计 PlantVillage / PlantDoc 番茄三类样本数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--root", action="append", type=parse_root_arg, required=False,
                    metavar="NAME=PATH",
                    help="可重复，例如 --root plantvillage=D:/data/.../color "
                         "--root plantdoc=D:/data/PlantDoc-Dataset")
    ap.add_argument("--mapping", type=Path, default=None,
                    help="自定义类名映射 JSON（可选）")
    ap.add_argument("--out-csv", type=Path, default=Path("stats/class_counts.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("stats/class_counts.md"))
    ap.add_argument("--list-only", action="store_true",
                    help="只列出目录里的真实类名，不写文件（第一次运行请先用这个）")
    ap.add_argument("--include-variants", action="store_true",
                    help="把 grayscale / segmented 变体也算进来（默认不算）")
    ap.add_argument("--dump-mapping-template", action="store_true",
                    help="打印当前使用的映射模板 JSON 后退出")
    args = ap.parse_args(argv)

    if args.dump_mapping_template:
        print(json.dumps(DEFAULT_MAP, ensure_ascii=False, indent=2))
        return 0

    if not args.root:
        ap.error("至少需要一个 --root NAME=PATH")

    mapping = load_mapping(args.mapping)
    skip = set() if args.include_variants else SKIP_VARIANTS_BY_DEFAULT

    rows, unmatched, skipped = collect(args.root, mapping, skip, args.list_only)

    if args.list_only:
        print("\n[提示] 请用上面的真实类名核对 data_stats.py 里的 DEFAULT_MAP，"
              "或写一个 --mapping JSON。")
        return 0

    if not rows:
        print("[ERROR] 没有统计到任何图片，未写文件。", file=sys.stderr)
        return 2

    note = "；".join(f"{n}={p}" for n, p in args.root)
    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md, unmatched, skipped, note)

    total = sum(r["n_images"] for r in rows)
    print(f"\n总计扫描到 {total} 张图片。")
    if unmatched:
        print(f"[WARN] 有 {len(unmatched)} 个目录未匹配统一标签（见 Markdown 末尾），"
              f"请先核对再写进报告。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
