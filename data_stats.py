#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_stats.py
=============
Count images per class for the three tomato classes (healthy / early_blight / late_blight)
in PlantVillage and PlantDoc, and write the result as CSV and Markdown.

Design notes
--------
1. No hard-coded layout: the script walks the tree and treats any folder that contains
   image files as one class folder; its parent name (train / test / color) becomes the split.
2. DEFAULT_MAP is only a draft: check the real folder names before relying on it.
   Unmatched folder names are printed, so naming differences in PlantDoc are visible.
3. Read-only: no image is modified or moved, and a corrupt file is counted, not fatal.

Usage
----
    # 1) list the real class names first (no counting)
    python data_stats.py --root plantvillage=<path>/PlantVillage-Dataset/raw/color \
                         --root plantdoc=<path>/PlantDoc-Dataset --list-only

    # 2) count and write CSV + Markdown
    python data_stats.py --root plantvillage=<path>/PlantVillage-Dataset/raw/color \
                         --root plantdoc=<path>/PlantDoc-Dataset \
                         --out-csv stats/class_counts.csv \
                         --out-md  stats/class_counts.md

    # 3) supply your own mapping (JSON; see --dump-mapping-template)
    python data_stats.py ... --mapping my_mapping.json

Output
----
* stats/class_counts.csv   machine readable, easy to load in Excel
* stats/class_counts.md    ready to paste into the report
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
# Constants
# --------------------------------------------------------------------------- #
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

UNIFIED_LABELS = ["healthy", "early_blight", "late_blight"]

# Display names for the unified labels
LABEL_DISPLAY = {
    "healthy": "healthy",
    "early_blight": "early_blight",
    "late_blight": "late_blight",
}

# split / variant names: used as a column when they appear in the path
SPLIT_NAMES = {"train", "test", "val", "valid", "validation", "training", "testing"}
VARIANT_NAMES = {"color", "grayscale", "segmented"}
# Non-colour variants are skipped by default: the PlantVillage repository ships
# color / grayscale / segmented copies, and counting all three triples the image count
SKIP_VARIANTS_BY_DEFAULT = {"grayscale", "segmented"}

# --------------------------------------------------------------------------- #
# Draft label mapping. Check the real folder names before using it.
# --------------------------------------------------------------------------- #
# PlantVillage uses "<Crop>___<Disease>" names; the three tomato classes are:
#   Tomato___healthy / Tomato___Early_blight / Tomato___Late_blight
# PlantDoc uses free-text folder names that differ between forks and versions,
# so treat the entries below as a starting point and run --list-only to confirm.
DEFAULT_MAP = {
    "plantvillage": {
        "Tomato___healthy": "healthy",
        "Tomato___Early_blight": "early_blight",
        "Tomato___Late_blight": "late_blight",
        # Some mirrors use spaces instead of underscores; kept as a fallback:
        "Tomato_healthy": "healthy",
        "Tomato_Early_blight": "early_blight",
        "Tomato_Late_blight": "late_blight",
        "Tomato healthy": "healthy",
        "Tomato Early blight": "early_blight",
        "Tomato Late blight": "late_blight",
    },
    "plantdoc": {
        # Verified against the train/ listing of the GitHub repository (master, 2026-11):
        # PlantDoc has nine tomato folders; these three are the ones this project uses.
        # Note: PlantDoc has no "healthy" label; the healthy class is called "Tomato leaf".
        "Tomato leaf": "healthy",
        "Tomato Early blight leaf": "early_blight",
        "Tomato leaf late blight": "late_blight",
        # Spellings seen in other forks and older mirrors; harmless if they do not match.
        "Tomato healthy": "healthy",
        "Tomato leaf healthy": "healthy",
        "Tomato Early blight": "early_blight",
        "Tomato Late blight leaf": "late_blight",
        "Tomato late blight leaf": "late_blight",
    },
}


# --------------------------------------------------------------------------- #
# Directory scanning
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
    Walk root and return [(split, class_name, dir, n_images)] for every folder that holds images.

    * split   : train / test / val / color when present in the path, otherwise "all"
    * class   : the folder name, i.e. the original class label
    * folders skipped as grayscale / segmented are returned separately

    Edge case: the dataset root may itself contain an image or two (for example
    PlantDoc_Examples.png). If real class folders exist below it, the root is not counted
    as a class, otherwise it would show up as an unmatched class and confuse the report.
    If the user passes a class folder as --root and it has no sub-folders with images,
    that root is treated as the class itself.
    """
    found, skipped = [], []
    root = root.resolve()
    root_hit = None
    for cur_dir, sub_dirs, files in os.walk(root):
        # skip hidden folders such as .git and .ipynb_checkpoints
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
        found.append(root_hit)      # the user pointed at a class folder directly
    return found, skipped


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #
def load_mapping(path: Path | None) -> dict:
    """Load a custom mapping file, falling back to DEFAULT_MAP."""
    if path is None:
        return {k: dict(v) for k, v in DEFAULT_MAP.items()}
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    merged = {k: dict(v) for k, v in DEFAULT_MAP.items()}
    for ds, m in user.items():
        merged.setdefault(ds, {}).update(m)
    return merged


def normalize_class_name(name: str) -> str:
    """Normalise a name before matching: lower case, strip separators, collapse spaces."""
    s = name.strip().lower().replace("___", " ").replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def build_lookup(ds_mapping: dict) -> dict:
    return {normalize_class_name(k): v for k, v in ds_mapping.items()}


# --------------------------------------------------------------------------- #
# Main counting routine
# --------------------------------------------------------------------------- #
def collect(roots: list[tuple[str, Path]], mapping: dict, skip_variants: set[str],
            list_only: bool = False):
    rows = []            # one entry per (dataset, split, raw_class)
    unmatched = []       # (dataset, split, raw_class, n) that the mapping does not cover
    skipped_all = []

    for ds_name, root in roots:
        if not root.exists():
            print(f"[WARN] dataset path not found, skipped: {ds_name} -> {root}", file=sys.stderr)
            continue
        ds_map = mapping.get(ds_name, {})
        lookup = build_lookup(ds_map)
        found, skipped = scan_class_dirs(root, skip_variants)
        skipped_all.extend([(ds_name, *s) for s in skipped])
        if not found:
            print(f"[WARN] no folder with images was found under {root}; "
                  f"check whether the archive was extracted one level deeper, e.g. */raw/color.",
                  file=sys.stderr)
        # the same (split, class) can appear in several folders; merge the counts
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
                hit = lookup.get(normalize_class_name(cls), "-- unmatched --")
                print(f"  [{split:9s}] {cls:38s} n={n:6d}  ->  {hit}")
    return rows, unmatched, skipped_all


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate rows by (dataset, unified_label), keeping mapped rows only."""
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
    # utf-8-sig keeps Excel from mangling non-ASCII text
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[OK] CSV written: {out_path}")


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
    md.append("# Dataset statistics (generated)\n")
    md.append(f"- Generated by `data_stats.py`\n- Image files only; extensions: "
              f"{sorted(IMAGE_EXTS)}\n")
    if note:
        md.append(f"- Data roots: {note}\n")

    datasets = list(OrderedDict.fromkeys(r["dataset"] for r in rows))
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] == ds]
        total = sum(r["n_images"] for r in sub)
        md.append(f"\n\n## {ds} ({total} images)\n")
        body = [[r["split"], r["raw_class"], r["unified_label"] or "**[unmatched]**",
                 r["n_images"]] for r in sub]
        md.append(markdown_table(["split", "original class", "unified label", "images"], body))
        md.append(f"\n\n> Subtotal: **{total}** images, of which "
                  f"**{sum(r['n_images'] for r in sub if r['unified_label'])}** are mapped.\n")

    md.append("\n\n## Summary per unified label (Table 1 of the report)\n")
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
    body.append(["Total"] + totals)      # one total per column, matching the header
    md.append(markdown_table(["unified label"] + datasets, body))

    if unmatched:
        md.append("\n\n## Unmatched folders (check these by hand)\n")
        md.append(markdown_table(["dataset", "split", "original class", "images"],
                                 [[u[0], u[1], u[2], u[3]] for u in unmatched]))
    if skipped:
        md.append("\n\n## Skipped non-colour variants (not counted)\n")
        md.append(markdown_table(["dataset", "folder", "images"],
                                 [[s[0], s[1], s[2]] for s in skipped]))
    md.append("\n\n## Still to fill in\n")
    md.append("- [TODO: check the numbers and paste this table into Table 1, Section 3.1]\n")
    md.append("- [TODO: state how many corrupt or non-leaf images were removed]\n")
    md.append("- [TODO: describe the near-duplicate check and its result]\n")

    out_path.write_text("".join(md), encoding="utf-8")
    print(f"[OK] Markdown written: {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_root_arg(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            "expected NAME=PATH, e.g. plantvillage=D:/data/PlantVillage/color")
    name, path = s.split("=", 1)
    return name.strip().lower(), Path(path.strip().strip('"'))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Count images per class for the three tomato classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--root", action="append", type=parse_root_arg, required=False,
                    metavar="NAME=PATH",
                    help="repeatable, e.g. --root plantvillage=D:/data/.../color "
                         "--root plantdoc=D:/data/PlantDoc-Dataset")
    ap.add_argument("--mapping", type=Path, default=None,
                    help="custom label mapping as JSON (optional)")
    ap.add_argument("--out-csv", type=Path, default=Path("stats/class_counts.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("stats/class_counts.md"))
    ap.add_argument("--list-only", action="store_true",
                    help="only list the real class names, write nothing (start here)")
    ap.add_argument("--include-variants", action="store_true",
                    help="also count grayscale / segmented variants (off by default)")
    ap.add_argument("--dump-mapping-template", action="store_true",
                    help="print the mapping template as JSON and exit")
    args = ap.parse_args(argv)

    if args.dump_mapping_template:
        print(json.dumps(DEFAULT_MAP, ensure_ascii=False, indent=2))
        return 0

    if not args.root:
        ap.error("at least one --root NAME=PATH is required")

    mapping = load_mapping(args.mapping)
    skip = set() if args.include_variants else SKIP_VARIANTS_BY_DEFAULT

    rows, unmatched, skipped = collect(args.root, mapping, skip, args.list_only)

    if args.list_only:
        print("\n[NOTE] check DEFAULT_MAP in data_stats.py against the names above, "
              "or pass your own --mapping JSON.")
        return 0

    if not rows:
        print("[ERROR] no images found, nothing written.", file=sys.stderr)
        return 2

    note = "；".join(f"{n}={p}" for n, p in args.root)
    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md, unmatched, skipped, note)

    total = sum(r["n_images"] for r in rows)
    print(f"\nScanned {total} images in total.")
    if unmatched:
        print(f"[WARN] {len(unmatched)} folders have no unified label (listed at the end "
              f"of the Markdown); check them before writing the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
