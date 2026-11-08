#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
===============
Build the two figures used in the report:

1. sample_images.png      two random images per class in a grid (3 rows x 2 columns)
2. class_distribution.png bar chart of images per class (both datasets side by side)

Requires matplotlib and Pillow, and reuses the scanning code in data_stats.py.

Usage
----
    python make_figures.py --root plantvillage=D:/data/PlantVillage/color \
                           --root plantdoc=D:/data/PlantDoc-Dataset \
                           --out-dir figures --seed 0

    # PlantVillage only
    python make_figures.py --root plantvillage=D:/data/PlantVillage/color

Notes
----
* The sampled file names are not hard-coded; the seed is fixed, so re-runs match.
  The sampled files are listed in figures/figure_sources.txt for the figure caption.
* All labels are in English so that a missing CJK font cannot turn them into boxes.
* Counts come from the actual scan; quote the script output, never hand-written numbers.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from data_stats import (  # reuse the scanning logic so both scripts agree
    DEFAULT_MAP,
    LABEL_DISPLAY,
    UNIFIED_LABELS,
    build_lookup,
    load_mapping,
    normalize_class_name,
    scan_class_dirs,
)

# Short English names for the unified labels
SHORT = {"healthy": "Healthy", "early_blight": "Early blight", "late_blight": "Late blight"}
COLORS = {"healthy": "#4C9F70", "early_blight": "#E0A458", "late_blight": "#B23A48"}


def parse_root_arg(s: str):
    if "=" not in s:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = s.split("=", 1)
    return name.strip().lower(), Path(path.strip().strip('"'))


def collect_images(roots, mapping, max_per_class_split="all"):
    """
    Return data[dataset][unified_label] = [Path, ...].
    A class spread over train/ and test/ is merged into one list.
    unmatched is deduplicated by class name so train/ and test/ are not reported twice.
    """
    data, unmatched = {}, {}
    for ds_name, root in roots:
        if not root.exists():
            print(f"[WARN] path not found, skipped: {ds_name} -> {root}", file=sys.stderr)
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
    """Pick k random images that PIL can actually open."""
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
            print(f"[WARN] skipping unreadable image {p} ({e})", file=sys.stderr)
    return out


def make_sample_grid(data, ds_name, out_path, per_class=2, seed=0, layout="3x2"):
    rng = random.Random(seed)
    bucket = data[ds_name]
    # Build the tiles: grouped by class (3 x 2) or by sample index (2 x 3)
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
        """Hide ticks but keep the axes, otherwise the xlabel with the file name is dropped."""
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
                print(f"[WARN] failed to open {p} ({e})", file=sys.stderr)
                continue
            ax.imshow(img)
        # Class name in the title, file name as the x label, so they cannot overlap
        ax.set_title(SHORT[lab], fontsize=10, pad=3)
        if p is not None:
            ax.set_xlabel(p.name[:26], fontsize=6, color="#555", labelpad=1)

    fig.suptitle(f"{ds_name} — tomato leaf classes "
                 f"({len(UNIFIED_LABELS)} classes × {per_class} random samples, seed={seed})",
                 fontsize=11)
    # Manual spacing keeps the figure compact enough for one page
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.03,
                        wspace=0.06, hspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] sample grid saved: {out_path}")
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
    # With more than one dataset the legend uses a grey patch to explain solid vs hatched,
    # instead of borrowing a class colour that would look like a class legend.
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
    print(f"[OK] class distribution saved: {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the sample grid and the class distribution chart")
    ap.add_argument("--root", action="append", type=parse_root_arg, required=True,
                    metavar="NAME=PATH", help="repeatable: --root plantvillage=... --root plantdoc=...")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--mapping", type=Path, default=None)
    ap.add_argument("--per-class", type=int, default=2, help="samples per class (default 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=["3x2", "2x3"], default="3x2",
                    help="3x2 = 3 rows by 2 columns (default); 2x3 = 2 rows by 3 columns")
    ap.add_argument("--grid-source", default=None,
                    help="dataset used for the sample grid (default: the first --root)")
    args = ap.parse_args(argv)

    mapping = load_mapping(args.mapping)
    data, unmatched = collect_images(args.root, mapping)
    if not data:
        print("[ERROR] no images were read; check the --root paths.", file=sys.stderr)
        return 2
    if unmatched:
        # Report the ignored classes with their counts; this is expected (other crops in PlantDoc)
        print(f"[WARN] {len(unmatched)} class folders have no unified label and are ignored:"
              + "；".join(f"{c} (n={n})" for (_d, c), n in unmatched[:6])
              + ("…" if len(unmatched) > 6 else ""))
        print("       to include them, edit DEFAULT_MAP in data_stats.py or pass --mapping.")

    for ds, bucket in data.items():
        print(f"[INFO] {ds}: " + ", ".join(f"{lab}={len(v)}" for lab, v in bucket.items()))

    grid_ds = (args.grid_source or list(data.keys())[0]).lower()
    if grid_ds not in data:
        print(f"[WARN] --grid-source={grid_ds} was not loaded; using {list(data.keys())[0]}")
        grid_ds = list(data.keys())[0]

    sources = make_sample_grid(data, grid_ds, args.out_dir / "sample_images.png",
                               per_class=args.per_class, seed=args.seed, layout=args.layout)
    make_class_distribution(data, args.out_dir / "class_distribution.png")

    # Record which files were used, for the caption and for re-runs
    src_file = args.out_dir / "figure_sources.txt"
    with open(src_file, "w", encoding="utf-8") as f:
        f.write(f"# dataset={grid_ds} seed={args.seed} per_class={args.per_class} "
                f"layout={args.layout}\n")
        for lab, paths in sources.items():
            for p in paths:
                f.write(f"{lab}\t{p}\n")
    print(f"[OK] figure sources saved: {src_file}")
    print("[NOTE] caption: two random samples per class (seed above); "
          "counts are in class_distribution.png and the data_stats.py output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
