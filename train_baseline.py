#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_baseline.py
=================
Minimal training script for three-class tomato leaf classification.

Two models are supported:
  * basiccnn  : three conv blocks, global average pooling, one linear layer (94k parameters)
  * resnet18  : torchvision ResNet18, optional ImageNet weights, three-class head

Experiment set (E1-E5 in Section 3.4 of the report):
  E1  --model basiccnn
  E2  --model resnet18
  E3  --model resnet18 --pretrained
  E4  --model resnet18 --pretrained --augment
  E5  reuses the E4 checkpoint for cross-domain evaluation; no retraining
      (use --eval-only with the E4 best.pt, see the README)

Examples
--------
    # 0) check the data before PyTorch is installed (PIL and numpy only)
    python train_baseline.py --data-root D:/data/PlantVillage/color --dry-run

    # 1) print the model summary and parameter count
    python train_baseline.py --model resnet18 --print-model

    # 2) train the baseline (CPU is fine; start with a small subset and few epochs)
    python train_baseline.py --data-root D:/data/PlantVillage/color \
        --model basiccnn --epochs 5 --batch-size 32 --limit-per-class 200 \
        --out-dir runs/E1_basiccnn_cpu

    # 3) transfer learning with augmentation
    python train_baseline.py --data-root D:/data/PlantVillage/color \
        --model resnet18 --pretrained --augment --epochs 10 \
        --out-dir runs/E4_resnet18_pre_aug

    # 4) cross-domain evaluation with a trained checkpoint (no retraining)
    python train_baseline.py --data-root D:/data/PlantVillage/color \
        --model resnet18 --pretrained --augment --eval-only \
        --ckpt runs/E4_resnet18_pre_aug/best.pt \
        --plantdoc-root D:/data/PlantDoc-Dataset --split test

Output (inside --out-dir)
--------------------
    log.csv          train_loss / train_acc / val_loss / val_acc / val_macro_f1 per epoch
    args.json        every argument of the run
    best.pt/last.pt  weights
    metrics_val.json / metrics_test.json / metrics_plantdoc.json
    confusion_matrix_test.csv
    predictions_test.csv  per-image predictions, for error analysis
    train.log        a copy of stdout, produced by redirection (see README)

Notes
----
* No accuracy is assumed anywhere; every number comes from a real run on your machine.
* Class weighting is off by default; --class-weight inverse enables inverse-frequency
    weighting, which is an extra experiment and must be reported separately.
* --num-workers defaults to 0 on Windows to avoid DataLoader handle problems.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from data_stats import (
    DEFAULT_MAP,
    UNIFIED_LABELS,
    build_lookup,
    load_mapping,
    normalize_class_name,
    scan_class_dirs,
)

# PyTorch is only needed for training; --dry-run works without it
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    HAS_TORCH = True
except ModuleNotFoundError:  # --dry-run still works without torch
    torch = None
    nn = None
    DataLoader = Dataset = None
    models = transforms = None
    HAS_TORCH = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = list(UNIFIED_LABELS)  # fixed label order: healthy, early blight, late blight


# ==========================================================================
# 1. Arguments
# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Three-class tomato leaf training: baseline and transfer learning",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    ap.add_argument("--data-root", type=Path, required=False,
                    help="root of the PlantVillage colour images (class sub-folders inside)")
    ap.add_argument("--dataset-name", default="plantvillage",
                    help="key used to look up the label mapping (default plantvillage)")
    ap.add_argument("--mapping", type=Path, default=None, help="custom label mapping JSON")
    ap.add_argument("--plantdoc-root", type=Path, default=None,
                    help="optional: PlantDoc root for the cross-domain evaluation")
    ap.add_argument("--plantdoc-split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="cap per class (0 = all images); use 50 for a smoke test")
    ap.add_argument("--split", default="70/15/15",
                    help="train/validation/test split (default 70/15/15)")

    # Model
    ap.add_argument("--model", default="basiccnn", choices=["basiccnn", "resnet18"])
    ap.add_argument("--pretrained", action="store_true",
                    help="start from ImageNet weights (downloaded once, then cached)")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="freeze conv1 to layer1 of ResNet18; faster on small data and on CPU")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--print-model", action="store_true", help="print the model summary and exit")

    # Training
    ap.add_argument("--augment", action="store_true", help="enable training-time augmentation")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate for the head or the CNN")
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="learning rate for the pretrained backbone (default --lr / 10)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--class-weight", default="none", choices=["none", "inverse"],
                    help="inverse = weight classes by inverse frequency (extra experiment, off by default)")
    ap.add_argument("--patience", type=int, default=5,
                    help="stop after this many epochs without a validation macro-F1 gain (0 = never)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap the number of steps per epoch (0 = no cap), useful for a quick check")

    # Run control
    ap.add_argument("--out-dir", type=Path, default=Path("runs/baseline"))
    ap.add_argument("--eval-only", action="store_true", help="evaluate only, no training")
    ap.add_argument("--ckpt", type=Path, default=None, help="checkpoint used by --eval-only")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the data and the configuration only, no training (no PyTorch needed)")
    return ap.parse_args(argv)


# ==========================================================================
# 2. Data indexing (no torch needed)
# ==========================================================================
def build_index(root: Path, ds_name: str, mapping: dict,
                limit_per_class: int = 0, split_filter: str = "all"):
    """
    Return items: [(Path, label:int)] with labels ordered as CLASS_NAMES.
    A statistics dictionary is returned as well.
    """
    if not root.exists():
        raise FileNotFoundError(f"data directory not found: {root}")
    lookup = build_lookup(mapping.get(ds_name, DEFAULT_MAP.get(ds_name, {})))
    found, skipped = scan_class_dirs(root, {"grayscale", "segmented"})

    per_class: dict[int, list[Path]] = {i: [] for i in range(len(CLASS_NAMES))}
    unmatched = []
    for split_name, cls, d, _n in found:
        if split_filter != "all" and split_name not in (split_filter, "all"):
            continue
        lab = lookup.get(normalize_class_name(cls))
        if lab is None:
            unmatched.append((split_name, cls, _n))
            continue
        idx = CLASS_NAMES.index(lab)
        per_class[idx].extend(sorted(p for p in d.iterdir()
                                     if p.is_file() and p.suffix.lower() in IMG_EXTS))

    rng = random.Random(0)
    items = []
    for idx, paths in per_class.items():
        if limit_per_class and len(paths) > limit_per_class:
            rng.shuffle(paths)
            paths = sorted(paths[:limit_per_class])
        items.extend((p, idx) for p in paths)
    stats = {
        "root": str(root.resolve()),
        "per_class": {CLASS_NAMES[i]: len(v) for i, v in per_class.items()},
        "unmatched_dirs": unmatched,
        "skipped_variants": skipped,
        "total": len(items),
    }
    return items, stats


def stratified_split(items, ratios=(0.70, 0.15, 0.15), seed=42):
    """Class-stratified split returning (train, val, test); every class reaches val and test."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "the split ratios must sum to 1"
    rng = random.Random(seed)
    by_cls: dict[int, list] = {}
    for p, y in items:
        by_cls.setdefault(y, []).append((p, y))

    train, val, test = [], [], []
    for y, lst in sorted(by_cls.items()):
        lst = list(lst)
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # Tiny-class guard: keep at least one image in val and test
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            n_train = n - n_val - n_test
        else:
            n_test = n - n_train - n_val
        n_train = max(1, n_train)
        train += lst[:n_train]
        val += lst[n_train:n_train + n_val]
        test += lst[n_train + n_val:]
    rng.shuffle(train)
    return train, val, test


def describe_split(name, split):
    cnt = {c: 0 for c in CLASS_NAMES}
    for _p, y in split:
        cnt[CLASS_NAMES[y]] += 1
    print(f"  {name:<6s} n={len(split):5d}  " +
          "  ".join(f"{k}={v}" for k, v in cnt.items()))
    return cnt


# ==========================================================================
# 3. Metrics (numpy only, no sklearn)
# ==========================================================================
def confusion_matrix(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_scores(cm):
    """Return [(support, precision, recall, f1)] in row order."""
    out = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append((support, float(prec), float(rec), float(f1)))
    return out


def macro_f1(cm):
    """Average over classes with support > 0, so an absent class is not scored as zero."""
    scores = [s for s in per_class_scores(cm) if s[0] > 0]
    if not scores:
        return 0.0
    return float(np.mean([s[3] for s in scores]))


# ==========================================================================
# 4. Models (torch required)
# ==========================================================================
if HAS_TORCH:

    class BasicCNN(nn.Module):
        """
        Three-block convolutional baseline:
          Conv3-32  -> BN -> ReLU -> MaxPool   (224 -> 112)
          Conv32-64 -> BN -> ReLU -> MaxPool   (112 -> 56)
          Conv64-128-> BN -> ReLU -> MaxPool   (56 -> 28)
          AdaptiveAvgPool(1) -> Dropout -> FC(128 -> 3)
        Small and simple: it answers what is possible without any transfer or augmentation.
        """

        def __init__(self, num_classes=3, dropout=0.3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, num_classes))

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)
            return self.classifier(x)

    class ImageListDataset(Dataset):
        """Wrap [(path, label)] as a Dataset; unreadable images become a black image and are counted."""

        def __init__(self, items, transform, verify=False):
            self.items = items
            self.transform = transform
            self.verify = verify
            self.bad = 0

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            from PIL import Image
            path, y = self.items[i]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:  # noqa: BLE001 corrupt image fallback
                self.bad += 1
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            return self.transform(img), y


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(name, num_classes=3, pretrained=False, freeze_backbone=False):
    """Build a model and return it with a short description; also used by --print-model."""
    if not HAS_TORCH:
        raise SystemExit("PyTorch is not installed; run `pip install -r requirements.txt`.")
    if name == "basiccnn":
        model = BasicCNN(num_classes=num_classes)
        return model, "BasicCNN (3 conv blocks, GAP, FC)"

    # torchvision compatibility: new versions take weights=, old ones take pretrained=
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        note = "ResNet18 (ImageNet pretrained)" if pretrained else "ResNet18 (random init)"
    except AttributeError:  # torchvision < 0.13
        model = models.resnet18(pretrained=pretrained)
        note = "ResNet18 (legacy API)"
    # Replace the classifier so that fc outputs three logits
    n_feat = model.fc.in_features
    model.fc = nn.Linear(n_feat, num_classes)
    if freeze_backbone:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        note += " [conv1+bn1+layer1 frozen]"
    return model, note


# ==========================================================================
# 5. transforms
# ==========================================================================
def build_transforms(image_size=224, augment=False):
    if not HAS_TORCH:
        raise SystemExit("PyTorch is not installed.")
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Without augmentation the training set still needs the same geometry as validation,
        train_tf = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


# ==========================================================================
# 6. Training and evaluation
# ==========================================================================
def evaluate(model, loader, criterion, device, num_classes=3, collect=None):
    model.eval()
    total_loss, n, correct = 0.0, 0, 0
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(1)
            n += y.size(0)
            correct += (pred == y).sum().item()
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            if collect is not None:
                pr = torch.softmax(logits, 1).cpu().numpy()
                probs.extend(pr.tolist())
    cm = confusion_matrix(ys, ps, num_classes)
    return {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(n, 1),
        "macro_f1": macro_f1(cm),
        "n": n,
        "cm": cm,
        "y_true": ys,
        "y_pred": ps,
        "probs": probs,
    }


def save_metrics(metrics, out_dir, tag):
    payload = {
        "tag": tag,
        "n": metrics["n"],
        "loss": round(metrics["loss"], 6),
        "accuracy": round(metrics["acc"], 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "per_class": [
            {"class": CLASS_NAMES[i], "support": s[0], "precision": round(s[1], 6),
             "recall": round(s[2], 6), "f1": round(s[3], 6)}
            for i, s in enumerate(per_class_scores(metrics["cm"]))
        ],
        "confusion_matrix": metrics["cm"].tolist(),
        "class_order": CLASS_NAMES,
        "note": "Produced by an actual run; do not edit these numbers by hand.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"metrics_{tag}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cm_path = out_dir / f"confusion_matrix_{tag}.csv"
    with open(cm_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + CLASS_NAMES)
        for i, row in enumerate(metrics["cm"]):
            w.writerow([CLASS_NAMES[i]] + list(row))
    print(f"[OK] {p}")
    print(f"[OK] {cm_path}")
    return payload


def train(args):
    device = (torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available())
              else torch.device("cpu" if args.device == "auto" else args.device))
    print(f"[INFO] device = {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    mapping = load_mapping(args.mapping)
    items, stats = build_index(args.data_root, args.dataset_name, mapping,
                               args.limit_per_class)
    if stats["unmatched_dirs"]:
        print("[WARN] these folders have no unified label and are ignored:")
        for s, c, n in stats["unmatched_dirs"]:
            print(f"        {s}/{c} (n={n})")
    if stats["total"] == 0:
        raise SystemExit("no images were read; check --data-root and the label mapping.")
    if args.max_steps or args.limit_per_class:
        print("[WARN] smoke-test mode: the subset or the step count is truncated on purpose.")
        print("       The accuracy and macro-F1 below only show that the code runs; ")
        print("       they are not a result. For the real runs, drop ")
        print("       --limit-per-class and --max-steps.")

    r = [float(x) for x in args.split.split("/")]
    r = [x / sum(r) for x in r]
    train_items, val_items, test_items = stratified_split(items, tuple(r), args.seed)
    print("[INFO] split:")
    describe_split("train", train_items)
    describe_split("val", val_items)
    describe_split("test", test_items)

    train_tf, eval_tf = build_transforms(args.image_size, args.augment)
    ds_train = ImageListDataset(train_items, train_tf)
    ds_val = ImageListDataset(val_items, eval_tf)
    ds_test = ImageListDataset(test_items, eval_tf)
    g = torch.Generator()
    g.manual_seed(args.seed)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=False, generator=g)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

    model, note = build_model(args.model, len(CLASS_NAMES), args.pretrained,
                              args.freeze_backbone)
    model.to(device)
    total, trainable = count_params(model)
    print(f"[INFO] model = {note}")
    print(f"[INFO] params total={total:,}  trainable={trainable:,}")

    # Loss (optional class weighting)
    if args.class_weight == "inverse":
        cnt = np.array([sum(1 for _p, y in train_items if y == i) for i in range(len(CLASS_NAMES))],
                       dtype=np.float64)
        cnt = np.maximum(cnt, 1.0)
        w = cnt.sum() / (len(cnt) * cnt)
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[INFO] class weights = {np.round(w, 4).tolist()}")
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=args.label_smoothing)

    # Optimizer: the pretrained backbone gets a smaller learning rate
    head_ids = {id(p) for n_, p in model.named_parameters()
                if p.requires_grad and ("fc" in n_ or "classifier" in n_)}
    head_params = [p for n_, p in model.named_parameters()
                   if p.requires_grad and id(p) in head_ids]
    # Compare by id(): `p not in head_params` would compare tensors element-wise
    # and raise, because a multi-element tensor cannot be used as a truth value
    back_params = [p for n_, p in model.named_parameters()
                   if p.requires_grad and id(p) not in head_ids]
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr / 10
    if args.pretrained and back_params:
        groups = [{"params": back_params, "lr": backbone_lr},
                  {"params": head_params, "lr": args.lr}]
        print(f"[INFO] param groups: backbone lr={backbone_lr:g}, head lr={args.lr:g}")
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": args.lr}]
    optimizer = torch.optim.Adam(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    use_amp = device.type == "cuda"
    scaler = None
    if use_amp:  # torch.amp on newer releases, torch.cuda.amp on older ones
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log_path = out_dir / "log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8-sig")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                    "val_macro_f1", "lr_head", "sec", "n_train", "n_val"])
    log_f.flush()

    best_f1, best_epoch, bad_epochs = -1.0, -1, 0
    best_path = out_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, run_n, run_correct = 0.0, 0, 0
        for step, (x, y) in enumerate(dl_train, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            run_loss += loss.item() * y.size(0)
            run_n += y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            if args.max_steps and step >= args.max_steps:
                break
        scheduler.step()

        tr_loss = run_loss / max(run_n, 1)
        tr_acc = run_correct / max(run_n, 1)
        vm = evaluate(model, dl_val, criterion, device, len(CLASS_NAMES))
        sec = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[E{epoch:03d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={vm['loss']:.4f} val_acc={vm['acc']:.4f} "
              f"val_macroF1={vm['macro_f1']:.4f} | {sec:.1f}s")
        log_w.writerow([epoch, round(tr_loss, 6), round(tr_acc, 6), round(vm["loss"], 6),
                        round(vm["acc"], 6), round(vm["macro_f1"], 6), lr_now,
                        round(sec, 2), run_n, vm["n"]])
        log_f.flush()

        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = vm["macro_f1"], epoch, 0
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES, "args": vars(args),
                        "epoch": epoch, "val_macro_f1": vm["macro_f1"]}, best_path)
            print(f"       -> new best (val macro-F1={best_f1:.4f}), saved {best_path.name}")
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                print(f"[INFO] early stop: no validation gain for {bad_epochs} epochs (patience={args.patience})")
                break

    log_f.close()
    torch.save({"model": args.model, "state_dict": model.state_dict(),
                "class_names": CLASS_NAMES, "args": vars(args), "epoch": "last"}, 
               out_dir / "last.pt")

    print(f"[INFO] training finished: best epoch={best_epoch}, best val macro-F1={best_f1:.4f}")
    # Evaluate the best checkpoint on the PlantVillage test split
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    tm = evaluate(model, dl_test, criterion, device, len(CLASS_NAMES), collect=True)
    print(f"[TEST] acc={tm['acc']:.4f} macro_f1={tm['macro_f1']:.4f} n={tm['n']}")
    save_metrics(tm, out_dir, "test")

    # Per-image predictions, for the error analysis
    with open(out_dir / "predictions_test.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "correct", "p_healthy", "p_early", "p_late"])
        for (p, y), yp, pr in zip(test_items, tm["y_pred"], tm["probs"]):
            w.writerow([str(p), CLASS_NAMES[y], CLASS_NAMES[yp], int(y == yp)]
                       + [round(x, 6) for x in pr])
    print(f"[OK] {out_dir / 'predictions_test.csv'}")

    # Cross-domain evaluation (E5)
    if args.plantdoc_root:
        cross_eval(model, args, criterion, device)

    print("\n[Next] copy the numbers from metrics_test.json into Section 4.1 of the report "
          "and put a screenshot of log.csv in the appendix.")
    return 0


def cross_eval(model, args, criterion, device):
    mapping = load_mapping(args.mapping)
    items, stats = build_index(args.plantdoc_root, "plantdoc", mapping,
                               split_filter=args.plantdoc_split)
    if stats["unmatched_dirs"]:
        print("[WARN] PlantDoc folders without a unified label:")
        for s, c, n in stats["unmatched_dirs"]:
            print(f"        {s}/{c} (n={n})")
    if not items:
        print("[WARN] no PlantDoc images were read; cross-domain evaluation skipped.")
        return
    ds = ImageListDataset(items, build_transforms(args.image_size, False)[1])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    m = evaluate(model, dl, criterion, device, len(CLASS_NAMES))
    print(f"[PLANTDOC] acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} n={m['n']}")
    save_metrics(m, args.out_dir, "plantdoc")


# ==========================================================================
# 7. dry-run / print-model
# ==========================================================================
def dry_run(args):
    mapping = load_mapping(args.mapping)
    items, stats = build_index(args.data_root, args.dataset_name, mapping,
                               args.limit_per_class)
    print("=" * 66)
    print(f"data root      : {stats['root']}")
    print(f"images found   : {stats['total']}")
    for k, v in stats["per_class"].items():
        print(f"   {k:<14s} {v}")
    if stats["unmatched_dirs"]:
        print("unmatched folders (check the label mapping):")
        for s, c, n in stats["unmatched_dirs"]:
            print(f"   [{s}] {c}  n={n}")
    if stats["skipped_variants"]:
        print(f"skipped {len(stats['skipped_variants'])} non-colour variant folders")
    r = [float(x) for x in args.split.split("/")]
    r = tuple(x / sum(r) for x in r)
    tr, va, te = stratified_split(items, r, args.seed)
    print("after the stratified 70/15/15 split:")
    describe_split("train", tr)
    describe_split("val", va)
    describe_split("test", te)
    # Check up to 20 images; datasets sometimes contain 0-byte or truncated files
    from PIL import Image
    sample = items[:20]
    print(f"readability check on the first {len(sample)} images:")
    bad = 0
    for p, y in sample:
        try:
            with Image.open(p) as im:
                print(f"   OK  {CLASS_NAMES[y]:<13s} {im.size[0]}x{im.size[1]} "
                      f"{im.mode:<4s} {p.name}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"   BAD {p} ({e})")
    print(f"readable {len(sample) - bad} / {len(sample)}"
          + ("  <-- some files cannot be read; they become black images, delete them first" if bad else ""))
    print("=" * 66)
    print("configuration:")
    for k in ["model", "pretrained", "augment", "epochs", "batch_size", "lr",
              "image_size", "seed", "device"]:
        print(f"   {k:<12s} {getattr(args, k)}")
    print(f"PyTorch        : {'installed ' + torch.__version__ if HAS_TORCH else 'not installed (--dry-run still works)'}")
    print("\n[OK] data pipeline check passed. These are real scan results; no training has run yet.")
    return 0


def print_model(args):
    if not HAS_TORCH:
        raise SystemExit("--print-model needs PyTorch.")
    model, note = build_model(args.model, len(CLASS_NAMES), args.pretrained,
                              args.freeze_backbone)
    total, trainable = count_params(model)
    print("=" * 74)
    print(f"model            : {note}")
    print(f"input            : [1, 3, {args.image_size}, {args.image_size}]")
    print(f"output classes   : {len(CLASS_NAMES)}  ({', '.join(CLASS_NAMES)})")
    print(f"total parameters : {total:,}")
    print(f"trainable        : {trainable:,}")
    print("=" * 74)
    print(f"{'layer':<38s}{'type':<20s}{'params':>12s}")
    print("-" * 74)
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 and name:
            p = sum(x.numel() for x in mod.parameters(recurse=False))
            if p == 0:      # skip layers without parameters, such as ReLU or pooling
                continue
            print(f"{name[:37]:<38s}{type(mod).__name__:<20s}{p:>12,}")
    print("-" * 74)
    print(f"{'TOTAL':<38s}{'':<20s}{total:>12,}")
    # One dummy forward pass confirms the number of outputs
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, args.image_size, args.image_size))
    print(f"dummy input [1,3,{args.image_size},{args.image_size}] -> output {tuple(out.shape)}")
    print("[NOTE] put TOTAL into the parameter row of Table 2 in the report.")
    return 0


# ==========================================================================
# 8. main
# ==========================================================================
def main(argv=None):
    args = parse_args(argv)
    if args.print_model:
        return print_model(args)
    if args.dry_run:
        if args.data_root is None:
            raise SystemExit("--dry-run needs --data-root (the PlantVillage colour folder)")
        return dry_run(args)
    if args.eval_only:
        if args.ckpt is None:
            raise SystemExit("--eval-only needs --ckpt (a trained best.pt)")
        if args.plantdoc_root is None and args.data_root is None:
            raise SystemExit("--eval-only needs --plantdoc-root (cross-domain) or --data-root (in-domain)")
        if not HAS_TORCH:
            raise SystemExit("evaluation also needs PyTorch: pip install -r requirements.txt")
        return eval_only(args)
    if not HAS_TORCH:
        raise SystemExit("PyTorch is not installed: pip install -r requirements.txt (or use --dry-run)")
    if args.data_root is None:
        raise SystemExit("training needs --data-root (the PlantVillage colour folder)")
    return train(args)


def eval_only(args):
    """Load a checkpoint and evaluate it on PlantDoc or on the test split; no updates."""
    device = (torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available())
              else torch.device("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # No need to download pretrained weights here; the state dict overwrites them
    model, note = build_model(ckpt.get("model", args.model), len(CLASS_NAMES),
                              pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print(f"[INFO] loaded {args.ckpt} ({note}); no training will run.")
    mapping = load_mapping(args.mapping)
    if args.plantdoc_root:
        # Cross-domain: the whole PlantDoc set is a test set
        items, _stats = build_index(args.plantdoc_root, "plantdoc", mapping,
                                    split_filter=args.plantdoc_split)
        eval_items, split_note = items, f"PlantDoc[{args.plantdoc_split}]"
        tag = "plantdoc"
    else:
        # In-domain: rebuild the same split as training and take the test part only;
        # otherwise training images would inflate the reported test accuracy.
        items, _stats = build_index(args.data_root, args.dataset_name, mapping,
                                    args.limit_per_class)
        r = [float(x) for x in args.split.split("/")]
        r = [x / sum(r) for x in r]
        _tr, _va, eval_items = stratified_split(items, tuple(r), args.seed)
        split_note = f"PlantVillage test split (split_seed={args.seed})"
        tag = "test"
    if not eval_items:
        raise SystemExit("the evaluation set is empty; check the paths and the label mapping.")
    print(f"[INFO] evaluation set = {split_note}, {len(eval_items)} images")
    ds = ImageListDataset(eval_items, build_transforms(args.image_size, False)[1])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss()
    m = evaluate(model, dl, criterion, device, len(CLASS_NAMES), collect=True)
    print(f"[EVAL] acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} n={m['n']}")
    save_metrics(m, args.out_dir, tag)

    # Per-image predictions, written as predictions_plantdoc.csv or predictions_test.csv
    # The error analysis in Fig. 6 needs them, especially cross-domain.
    pred_path = args.out_dir / f"predictions_{tag}.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "correct", "p_healthy", "p_early", "p_late"])
        for (p, y), yp, pr in zip(eval_items, m["y_pred"], m["probs"]):
            w.writerow([str(p), CLASS_NAMES[y], CLASS_NAMES[yp], int(y == yp)]
                       + [round(x, 6) for x in pr])
    print(f"[OK] {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
