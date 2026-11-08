#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig1.py
============
生成报告 Figure 1：2 行 × 3 列的样本对比图

    行 = 数据集（PlantVillage / PlantDoc），列 = 三类（healthy / early blight / late blight）
    每个小格下方标 (a)…(f)，左侧写数据集名与说明

只用 Pillow，不需要 matplotlib。图片全部来自 data/ 下的真实数据，每个单元格随机抽 1 张
（固定随机种子，重跑结果一致）。

用法：
    python make_fig1.py --out figures/fig1_samples.png --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from data_stats import (DEFAULT_MAP, UNIFIED_LABELS, build_lookup,
                        load_mapping, normalize_class_name, scan_class_dirs)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

CELL = 260          # 单元格边长（像素）
PAD = 18            # 单元格内边距
LABEL_H = 26        # 小格下方 "(a) Healthy" 的高度
ROW_LABEL_W = 150   # 左侧行标签宽度
MARGIN = 24


def find_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def collect(root: Path, ds_name: str, mapping: dict, per_class=1, seed=0):
    lookup = build_lookup(mapping.get(ds_name, DEFAULT_MAP.get(ds_name, {})))
    found, _ = scan_class_dirs(root, {"grayscale", "segmented"})
    buckets = {lab: [] for lab in UNIFIED_LABELS}
    for _split, cls, d, _n in found:
        lab = lookup.get(normalize_class_name(cls))
        if lab is None:
            continue
        buckets[lab].extend(sorted(p for p in d.iterdir()
                                   if p.is_file() and p.suffix.lower() in
                                   {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}))
    rng = random.Random(seed)
    picked = {}
    for lab, paths in buckets.items():
        pool = list(paths)
        rng.shuffle(pool)
        picked[lab] = pool[:per_class]
    return picked


def fit_square(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures/fig1_samples.png"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plantvillage", type=Path,
                    default=Path("data/PlantVillage-Dataset/raw/color"))
    ap.add_argument("--plantdoc", type=Path, default=Path("data/PlantDoc-Dataset"))
    args = ap.parse_args()

    mapping = load_mapping(None)
    sets = [
        ("PlantVillage", "Controlled\nimages", collect(args.plantvillage, "plantvillage", mapping, 1, args.seed)),
        ("PlantDoc", "Real-world\nimages", collect(args.plantdoc, "plantdoc", mapping, 1, args.seed)),
    ]

    cols, rows = len(UNIFIED_LABELS), len(sets)
    cell = CELL + PAD * 2
    W = MARGIN * 2 + ROW_LABEL_W + cols * cell
    H = MARGIN * 2 + 40 + rows * (cell + LABEL_H)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    f_head = find_font(19)
    f_row = find_font(16)
    f_row_small = find_font(13)
    f_cell = find_font(14)

    titles = {"healthy": "Healthy", "early_blight": "Early blight", "late_blight": "Late blight"}
    letters = "abcdef"

    # 列标题
    for c, lab in enumerate(UNIFIED_LABELS):
        x = MARGIN + ROW_LABEL_W + c * cell + cell // 2
        tw = draw.textlength(titles[lab], font=f_head)
        draw.text((x - tw / 2, MARGIN), titles[lab], fill="black", font=f_head)

    for r, (ds, note, picked) in enumerate(sets):
        y0 = MARGIN + 40 + r * (cell + LABEL_H)
        # 左侧行标签
        draw.text((MARGIN, y0 + cell // 2 - 22), ds, fill="black", font=f_row)
        draw.multiline_text((MARGIN, y0 + cell // 2 + 2), note, fill="#444444",
                            font=f_row_small, spacing=2)
        for c, lab in enumerate(UNIFIED_LABELS):
            x0 = MARGIN + ROW_LABEL_W + c * cell
            # 外框
            draw.rectangle([x0 + 4, y0 + 4, x0 + cell - 4, y0 + cell - 4],
                           outline="#CCCCCC", width=1)
            paths = picked.get(lab) or []
            if paths:
                try:
                    with Image.open(paths[0]) as im:
                        thumb = fit_square(im, CELL)
                    canvas.paste(thumb, (x0 + PAD, y0 + PAD))
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] 打不开 {paths[0]}: {e}")
            else:
                draw.text((x0 + PAD + 6, y0 + CELL // 2), "(no image)",
                          fill="#999999", font=f_cell)
            # 小格标注 (a) Healthy
            idx = r * cols + c
            sub = f"({letters[idx]}) {titles[lab]}"
            tw = draw.textlength(sub, font=f_cell)
            draw.text((x0 + cell / 2 - tw / 2, y0 + cell + 2), sub, fill="black", font=f_cell)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, dpi=(250, 250))
    print(f"[OK] {args.out}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
