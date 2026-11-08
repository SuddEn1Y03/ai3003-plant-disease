#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
===============
生成报告用的两张图：

1. sample_images.png      每类随机抽 2 张，拼成网格（默认 3 行 × 2 列 = 3 类 × 2 张）
2. class_distribution.png 三个统一标签的样本数柱状图（有 PlantDoc 时并排对比）

依赖：matplotlib、Pillow（另需同目录下的 data_stats.py 复用目录扫描逻辑）

用法
----
    python make_figures.py --root plantvillage=D:/data/PlantVillage/color \
                           --root plantdoc=D:/data/PlantDoc-Dataset \
                           --out-dir figures --seed 0

    # 只画 PlantVillage
    python make_figures.py --root plantvillage=D:/data/PlantVillage/color

注意
----
* 脚本**不会**把随机抽到的文件名写死，随机种子固定（--seed），所以重跑结果一致；
  抽到的文件清单会写进 figures/figure_sources.txt，方便在报告图注里注明来源。
* 图里所有的类别名用英文，避免服务器缺中文字体出现"豆腐块"。
* 图片数量来自实际扫描；图注里的数字请以运行输出为准，不要手写。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能跑
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from data_stats import (  # 复用统计脚本的逻辑，避免两套口径
    DEFAULT_MAP,
    LABEL_DISPLAY,
    UNIFIED_LABELS,
    build_lookup,
    load_mapping,
    normalize_class_name,
    scan_class_dirs,
)

# 统一标签的英文短名（画图用）
SHORT = {"healthy": "Healthy", "early_blight": "Early blight", "late_blight": "Late blight"}
COLORS = {"healthy": "#4C9F70", "early_blight": "#E0A458", "late_blight": "#B23A48"}


def parse_root_arg(s: str):
    if "=" not in s:
        raise argparse.ArgumentTypeError("格式应为 NAME=PATH")
    name, path = s.split("=", 1)
    return name.strip().lower(), Path(path.strip().strip('"'))


def collect_images(roots, mapping, max_per_class_split="all"):
    """
    返回 data[dataset][unified_label] = [Path, ...]
    同一个类如果分散在 train/test 里，会合并成一个列表。
    unmatched 里按类名去重并累加计数，避免 train/test 各报一次刷屏。
    """
    data, unmatched = {}, {}
    for ds_name, root in roots:
        if not root.exists():
            print(f"[WARN] 路径不存在，跳过：{ds_name} -> {root}", file=sys.stderr)
            continue
        lookup = build_lookup(mapping.get(ds_name, DEFAULT_MAP.get(ds_name, {})))
        found, _skipped = scan_class_dirs(root, {"grayscale", "segmented"})
        bucket = {lab: [] for lab in UNIFIED_LABELS}
        for _split, cls, d, n in found:
            lab = lookup.get(normalize_class_name(cls))
            if lab is None:
                key = (ds_name, cls)
                unmatched[key] = unmatched.get(key, 0) + n
                continue
            bucket[lab].extend(sorted(p for p in d.iterdir()
                                      if p.is_file() and p.suffix.lower()
                                      in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}))
        data[ds_name] = bucket
    return data, sorted(unmatched.items())


def pick_samples(paths, k, rng):
    """随机抽 k 张能被 PIL 打开的图片（跳过损坏文件）。"""
    if not paths:
        return []
    pool = list(paths)
    rng.shuffle(pool)
    out = []
    for p in pool:
        if len(out) >= k:
            break
        try:
            with Image.open(p) as im:
                im.verify()
            out.append(p)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] 跳过无法读取的图片 {p} ({e})", file=sys.stderr)
    return out


def make_sample_grid(data, ds_name, out_path, per_class=2, seed=0, layout="3x2"):
    rng = random.Random(seed)
    bucket = data[ds_name]
    # 组织成 tiles：默认按"类"分组（3 行 × 2 列），也可以按"样本序号"分组（2 行 × 3 列）
    picked = {lab: pick_samples(bucket[lab], per_class, rng) for lab in UNIFIED_LABELS}
    tiles = []           # [(label, path_or_None)]
    if layout == "3x2":
        n_rows, n_cols = len(UNIFIED_LABELS), per_class
        for lab in UNIFIED_LABELS:
            imgs = picked[lab]
            for i in range(per_class):
                tiles.append((lab, imgs[i] if i < len(imgs) else None))
    else:  # 2x3
        n_rows, n_cols = per_class, len(UNIFIED_LABELS)
        for i in range(per_class):
            for lab in UNIFIED_LABELS:
                imgs = picked[lab]
                tiles.append((lab, imgs[i] if i < len(imgs) else None))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.45 * n_rows),
                             squeeze=False)

    def clean_axes(ax):
        """隐藏刻度但保留坐标轴，这样 xlabel（文件名）才能显示出来。"""
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    for idx, (lab, p) in enumerate(tiles):
        ax = axes[idx // n_cols][idx % n_cols]
        clean_axes(ax)
        if p is None:
            ax.text(0.5, 0.5, "N/A\n(no image)", ha="center", va="center", fontsize=10)
            ax.set_facecolor("#EEEEEE")
        else:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 打开失败 {p} ({e})", file=sys.stderr)
                continue
            ax.imshow(img)
        # 类名放标题，文件名放 x 轴标签（避免和标题重叠）
        ax.set_title(SHORT[lab], fontsize=10, pad=3)
        if p is not None:
            ax.set_xlabel(p.name[:26], fontsize=6, color="#555", labelpad=1)

    fig.suptitle(f"{ds_name} — tomato leaf classes "
                 f"({len(UNIFIED_LABELS)} classes × {per_class} random samples, seed={seed})",
                 fontsize=11)
    # 手动排版，比 tight_layout 更紧凑（论文里一页放得下）
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.03,
                        wspace=0.06, hspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 样本网格已保存：{out_path}")
    return {lab: [str(x) for x in v] for lab, v in picked.items()}


def make_class_distribution(data, out_path, title="Class distribution"):
    from matplotlib.patches import Patch

    datasets = list(data.keys())
    labels = UNIFIED_LABELS
    counts = {ds: [len(data[ds][lab]) for lab in labels] for ds in datasets}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.8 / max(len(datasets), 1)
    x = range(len(labels))
    for i, ds in enumerate(datasets):
        xs = [xi - 0.4 + width / 2 + i * width for xi in x]
        bars = ax.bar(xs, counts[ds], width=width,
                      color=[COLORS[l] for l in labels],
                      alpha=0.95 if i == 0 else 0.45,
                      hatch=None if i == 0 else "//",
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, counts[ds]):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                        ha="center", va="bottom", fontsize=8)
            else:
                ax.text(b.get_x() + b.get_width() / 2, 5, "0", ha="center",
                        va="bottom", fontsize=8, color="red")

    ax.set_xticks(list(x))
    ax.set_xticklabels([SHORT[l] for l in labels])
    ax.set_ylabel("Number of images")
    ax.set_title(title if len(datasets) > 1 else f"{title} — {datasets[0]}")
    # 数据集多于一个时用灰色示例块解释"实心/斜线"分别代表哪个数据集，
    # 而不是直接拿某一类的颜色，否则读图的人会以为图例在说类别。
    if len(datasets) > 1:
        handles = [Patch(facecolor="#888888", alpha=0.95 if i == 0 else 0.45,
                         hatch=None if i == 0 else "//", edgecolor="black", label=ds)
                   for i, ds in enumerate(datasets)]
        ax.legend(handles=handles, frameon=False, title="dataset", fontsize=9,
                  title_fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    fig.text(0.01, 0.01, "counts from actual directory scan (see stats/class_counts.md)",
             fontsize=7, color="#555")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 类别分布图已保存：{out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成样本网格图与类别分布图")
    ap.add_argument("--root", action="append", type=parse_root_arg, required=True,
                    metavar="NAME=PATH", help="可重复：--root plantvillage=... --root plantdoc=...")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--mapping", type=Path, default=None)
    ap.add_argument("--per-class", type=int, default=2, help="每类抽几张（默认 2）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=["3x2", "2x3"], default="3x2",
                    help="3x2 = 3 行(类) × 2 列(样本)，默认；2x3 = 2 行 × 3 列")
    ap.add_argument("--grid-source", default=None,
                    help="用哪个数据集画样本网格，默认第一个 --root")
    args = ap.parse_args(argv)

    mapping = load_mapping(args.mapping)
    data, unmatched = collect_images(args.root, mapping)
    if not data:
        print("[ERROR] 没有读到任何图片，请检查 --root 路径。", file=sys.stderr)
        return 2
    if unmatched:
        # 只提示"被忽略的类"，并给出张数；这是正常现象（PlantDoc 里非三类番茄、其他作物）
        print(f"[WARN] 有 {len(unmatched)} 个类目录没有映射到统一标签，已忽略（不会出现在图里）："
              + "；".join(f"{c} (n={n})" for (_d, c), n in unmatched[:6])
              + ("…" if len(unmatched) > 6 else ""))
        print("       如需纳入，改 data_stats.py 的 DEFAULT_MAP 或用 --mapping 指定映射文件。")

    for ds, bucket in data.items():
        print(f"[INFO] {ds}: " + ", ".join(f"{lab}={len(v)}" for lab, v in bucket.items()))

    grid_ds = (args.grid_source or list(data.keys())[0]).lower()
    if grid_ds not in data:
        print(f"[WARN] --grid-source={grid_ds} 不在已读数据集中，改用 {list(data.keys())[0]}")
        grid_ds = list(data.keys())[0]

    sources = make_sample_grid(data, grid_ds, args.out_dir / "sample_images.png",
                               per_class=args.per_class, seed=args.seed, layout=args.layout)
    make_class_distribution(data, args.out_dir / "class_distribution.png")

    # 记录图源，方便写图注 / 复现
    src_file = args.out_dir / "figure_sources.txt"
    with open(src_file, "w", encoding="utf-8") as f:
        f.write(f"# dataset={grid_ds} seed={args.seed} per_class={args.per_class} "
                f"layout={args.layout}\n")
        for lab, paths in sources.items():
            for p in paths:
                f.write(f"{lab}\t{p}\n")
    print(f"[OK] 图源清单已保存：{src_file}")
    print("[提示] 图注请写：每类随机抽取 2 张（随机种子见上），"
          "样本数见 class_distribution.png 与 data_stats.py 输出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
