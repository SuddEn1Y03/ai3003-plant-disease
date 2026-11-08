#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_baseline.py
=================
番茄叶片三分类（healthy / early_blight / late_blight）的极简训练脚本。

支持两种模型：
  * basiccnn  : 3 层卷积 + 全局平均池化 + 全连接（参数量很小，CPU 也能跑）
  * resnet18  : torchvision ResNet18，可选 ImageNet 预训练权重，分类头替换为 3 类

支持的实验（对应报告 3.6 的 E1–E5）：
  E1  --model basiccnn
  E2  --model resnet18
  E3  --model resnet18 --pretrained
  E4  --model resnet18 --pretrained --augment
  E5  训练时与 E3/E4 相同，训练完加 --plantdoc-root 做跨域评估
      （E5 不重新训练：用 E4 的 best.pt + `--eval-only`，见 README）

用法示例
--------
    # 0) 不装 PyTorch 也能先检查数据（只依赖 PIL/numpy）
    python train_baseline.py --data-root D:/data/PlantVillage/color --dry-run

    # 1) 打印模型结构和参数量
    python train_baseline.py --model resnet18 --print-model

    # 2) 训练 baseline（CPU 可跑，先用小样本 + 少量 epoch 验证流程）
    python train_baseline.py --data-root D:/data/PlantVillage/color \
        --model basiccnn --epochs 5 --batch-size 32 --limit-per-class 200 \
        --out-dir runs/E1_basiccnn_cpu

    # 3) 迁移学习 + 数据增强
    python train_baseline.py --data-root D:/data/PlantVillage/color \
        --model resnet18 --pretrained --augment --epochs 10 \
        --out-dir runs/E4_resnet18_pre_aug

    # 4) 用训好的权重做 PlantDoc 跨域评估（不重新训练）
    python train_baseline.py --data-root D:/data/PlantVillage/color \
        --model resnet18 --pretrained --augment --eval-only \
        --ckpt runs/E4_resnet18_pre_aug/best.pt \
        --plantdoc-root D:/data/PlantDoc-Dataset --split test

输出（--out-dir 下）
--------------------
    log.csv          每个 epoch 的 train_loss / train_acc / val_loss / val_acc / val_macro_f1
    args.json        本次运行的完整参数（复现用）
    best.pt/last.pt  权重
    metrics_val.json / metrics_test.json / metrics_plantdoc.json
    confusion_matrix_test.csv
    predictions_test.csv  逐图预测，方便找错例
    train.log        标准输出的副本（重定向得到，见 README）

注意
----
* 本脚本**不预设**任何准确率。所有数字都来自你机器上的真实运行。
* 类别不平衡时默认不加权；--class-weight inverse 可以开启反频率加权（属于额外实验，
  需要在报告里单独说明，不要和主实验混在一起报）。
* Windows 上默认 --num-workers 0，避免多进程 DataLoader 的句柄问题。
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

# PyTorch 只在真正训练时才需要；--dry-run / --print-model 前也不会 import 失败
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    HAS_TORCH = True
except ModuleNotFoundError:  # 没装 torch 时仍然可以跑 --dry-run
    torch = None
    nn = None
    DataLoader = Dataset = None
    models = transforms = None
    HAS_TORCH = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = list(UNIFIED_LABELS)  # 标签顺序固定：0=healthy 1=early 2=late


# ==========================================================================
# 1. 参数
# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="番茄叶片三分类 baseline / 迁移学习训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # 数据
    ap.add_argument("--data-root", type=Path, required=False,
                    help="PlantVillage 彩色图的根目录（其下是各类别子目录）")
    ap.add_argument("--dataset-name", default="plantvillage",
                    help="用于查类名映射的键，默认 plantvillage")
    ap.add_argument("--mapping", type=Path, default=None, help="自定义类名映射 JSON")
    ap.add_argument("--plantdoc-root", type=Path, default=None,
                    help="可选：PlantDoc 根目录，训练完做跨域评估")
    ap.add_argument("--plantdoc-split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="每类最多用多少张（0=全部）；冒烟测试用 50")
    ap.add_argument("--split", default="70/15/15",
                    help="训练/验证/测试比例，默认 70/15/15")

    # 模型
    ap.add_argument("--model", default="basiccnn", choices=["basiccnn", "resnet18"])
    ap.add_argument("--pretrained", action="store_true",
                    help="ResNet18 使用 ImageNet 预训练权重（需要联网下载，或本地缓存）")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="冻结 ResNet18 的 conv1~layer1（只训练后面的层），小样本/CPU 时更快")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--print-model", action="store_true", help="打印结构和参数量后退出")

    # 训练
    ap.add_argument("--augment", action="store_true", help="开启训练集数据增强")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="分类头/CNN 的学习率")
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="预训练 ResNet18 主干的学习率，默认 = --lr / 10")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--class-weight", default="none", choices=["none", "inverse"],
                    help="inverse = 按类别频率的倒数加权（额外实验，默认关闭）")
    ap.add_argument("--patience", type=int, default=5,
                    help="验证集 macro-F1 连续多少个 epoch 不提升就早停（0=不早停）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="每个 epoch 最多多少 step（0=不限），快速验证用")

    # 运行控制
    ap.add_argument("--out-dir", type=Path, default=Path("runs/baseline"))
    ap.add_argument("--eval-only", action="store_true", help="只做评估，不训练")
    ap.add_argument("--ckpt", type=Path, default=None, help="--eval-only 时的权重路径")
    ap.add_argument("--dry-run", action="store_true",
                    help="只检查数据和配置，不训练（不需要 PyTorch）")
    return ap.parse_args(argv)


# ==========================================================================
# 2. 数据索引（不依赖 torch）
# ==========================================================================
def build_index(root: Path, ds_name: str, mapping: dict,
                limit_per_class: int = 0, split_filter: str = "all"):
    """
    返回 items: [(Path, label:int)]，label 顺序固定为 CLASS_NAMES。
    同时返回统计信息 dict。
    """
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在：{root}")
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
    """按类别分层划分，返回 (train, val, test)。保证每类尽量有样本进 val/test。"""
    assert abs(sum(ratios) - 1.0) < 1e-6, "划分比例之和必须为 1"
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
        # 小样本保护：尽量让 val/test 至少各 1 张
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
# 3. 指标（纯 numpy，不依赖 sklearn）
# ==========================================================================
def confusion_matrix(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_scores(cm):
    """返回 [(support, precision, recall, f1)]，类别顺序即行顺序。"""
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
    """只在 support>0 的类别上取平均（测试集缺类时不会凭空记 0 分）。"""
    scores = [s for s in per_class_scores(cm) if s[0] > 0]
    if not scores:
        return 0.0
    return float(np.mean([s[3] for s in scores]))


# ==========================================================================
# 4. 模型（需要 torch）
# ==========================================================================
if HAS_TORCH:

    class BasicCNN(nn.Module):
        """
        3 层卷积基线：
          Conv3-32  -> BN -> ReLU -> MaxPool   (224 -> 112)
          Conv32-64 -> BN -> ReLU -> MaxPool   (112 -> 56)
          Conv64-128-> BN -> ReLU -> MaxPool   (56 -> 28)
          AdaptiveAvgPool(1) -> Dropout -> FC(128 -> 3)
        结构简单、参数少，用来回答"不加任何技巧能到什么水平"。
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
        """把 [(path, label)] 包成 Dataset；图片读取失败时返回全黑图并计数。"""

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
            except Exception:  # noqa: BLE001 损坏图片兜底
                self.bad += 1
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            return self.transform(img), y


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(name, num_classes=3, pretrained=False, freeze_backbone=False):
    """构造模型；返回 (model, 说明文字)。print-model 时也会调用它。"""
    if not HAS_TORCH:
        raise SystemExit("没有安装 PyTorch，先 `pip install -r requirements.txt`。")
    if name == "basiccnn":
        model = BasicCNN(num_classes=num_classes)
        return model, "BasicCNN (3 conv blocks, GAP, FC)"

    # torchvision 版本兼容：新版本用 weights=，老版本用 pretrained=
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        note = "ResNet18 (ImageNet pretrained)" if pretrained else "ResNet18 (random init)"
    except AttributeError:  # torchvision < 0.13
        model = models.resnet18(pretrained=pretrained)
        note = "ResNet18 (legacy API)"
    # 分类头替换：fc 输出 3 类
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
        raise SystemExit("没有安装 PyTorch。")
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
        # 不开增强时也要保证和验证集同样的几何处理，否则两组实验不可比
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
# 6. 训练 / 评估
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
        "note": "所有数字来自本机实际运行，请勿手改。",
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
        print("[WARN] 这些目录没匹配到统一标签，已忽略：")
        for s, c, n in stats["unmatched_dirs"]:
            print(f"        {s}/{c} (n={n})")
    if stats["total"] == 0:
        raise SystemExit("没有读到任何图片，请检查 --data-root 和类名映射。")
    if args.max_steps or args.limit_per_class or args.epochs <= 2:
        print("[WARN] 冒烟/调试模式：样本数或 step 数被刻意压小，训练不充分。")
        print("       这次跑出来的 acc / macro-F1 只用于确认代码能跑通，")
        print("       不代表模型效果，绝对不要写进报告。正式实验去掉")
        print("       --limit-per-class 和 --max-steps，并把 --epochs 设到 20 左右。")

    r = [float(x) for x in args.split.split("/")]
    r = [x / sum(r) for x in r]
    train_items, val_items, test_items = stratified_split(items, tuple(r), args.seed)
    print("[INFO] 划分：")
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

    # 损失函数（可选类别加权）
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

    # 优化器：预训练骨干用更小的学习率
    head_ids = {id(p) for n_, p in model.named_parameters()
                if p.requires_grad and ("fc" in n_ or "classifier" in n_)}
    head_params = [p for n_, p in model.named_parameters()
                   if p.requires_grad and id(p) in head_ids]
    # 注意：这里必须用 id() 比较，直接写 `p not in head_params` 会触发
    # tensor 的逐元素 == 比较，多元素张量无法转成 bool，会直接报错
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
    if use_amp:  # torch>=2.3 用 torch.amp，老版本退回 torch.cuda.amp
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
            print(f"       -> 新的 best（val macro-F1={best_f1:.4f}），已保存 {best_path.name}")
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                print(f"[INFO] 早停：验证集连续 {bad_epochs} 个 epoch 没有提升（patience={args.patience}）")
                break

    log_f.close()
    torch.save({"model": args.model, "state_dict": model.state_dict(),
                "class_names": CLASS_NAMES, "args": vars(args), "epoch": "last"}, 
               out_dir / "last.pt")

    print(f"[INFO] 训练结束：best epoch={best_epoch}, best val macro-F1={best_f1:.4f}")
    # 用 best 权重在 PV test 上评估
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    tm = evaluate(model, dl_test, criterion, device, len(CLASS_NAMES), collect=True)
    print(f"[TEST] acc={tm['acc']:.4f} macro_f1={tm['macro_f1']:.4f} n={tm['n']}")
    save_metrics(tm, out_dir, "test")

    # 逐图预测，方便找错例
    with open(out_dir / "predictions_test.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "correct", "p_healthy", "p_early", "p_late"])
        for (p, y), yp, pr in zip(test_items, tm["y_pred"], tm["probs"]):
            w.writerow([str(p), CLASS_NAMES[y], CLASS_NAMES[yp], int(y == yp)]
                       + [round(x, 6) for x in pr])
    print(f"[OK] {out_dir / 'predictions_test.csv'}")

    # 跨域评估（E5）
    if args.plantdoc_root:
        cross_eval(model, args, criterion, device)

    print("\n[下一步] 把 log.csv 的最后一行 / metrics_test.json 里的真实数字填进报告 4.1 节，"
          "把 log.csv 截图放进附录。")
    return 0


def cross_eval(model, args, criterion, device):
    mapping = load_mapping(args.mapping)
    items, stats = build_index(args.plantdoc_root, "plantdoc", mapping,
                               split_filter=args.plantdoc_split)
    if stats["unmatched_dirs"]:
        print("[WARN] PlantDoc 未匹配目录：")
        for s, c, n in stats["unmatched_dirs"]:
            print(f"        {s}/{c} (n={n})")
    if not items:
        print("[WARN] PlantDoc 没读到图片，跳过跨域评估。")
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
    print(f"数据根目录 : {stats['root']}")
    print(f"识别样本数 : {stats['total']}")
    for k, v in stats["per_class"].items():
        print(f"   {k:<14s} {v}")
    if stats["unmatched_dirs"]:
        print("未匹配目录（请核对类名映射）：")
        for s, c, n in stats["unmatched_dirs"]:
            print(f"   [{s}] {c}  n={n}")
    if stats["skipped_variants"]:
        print(f"已跳过非彩色变体 {len(stats['skipped_variants'])} 个目录")
    r = [float(x) for x in args.split.split("/")]
    r = tuple(x / sum(r) for x in r)
    tr, va, te = stratified_split(items, r, args.seed)
    print("按 70/15/15 分层划分后：")
    describe_split("train", tr)
    describe_split("val", va)
    describe_split("test", te)
    # 抽检最多 20 张，确认图片能解码（数据集里偶尔会有 0 字节 / 截断文件）
    from PIL import Image
    sample = items[:20]
    print(f"抽检图片可读性（随机前 {len(sample)} 张）：")
    bad = 0
    for p, y in sample:
        try:
            with Image.open(p) as im:
                print(f"   OK  {CLASS_NAMES[y]:<13s} {im.size[0]}x{im.size[1]} "
                      f"{im.mode:<4s} {p.name}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"   BAD {p} ({e})")
    print(f"可读 {len(sample) - bad} / {len(sample)}"
          + ("  <-- 有文件读不出来，训练时会被替换成黑图，建议先删除" if bad else ""))
    print("=" * 66)
    print("配置：")
    for k in ["model", "pretrained", "augment", "epochs", "batch_size", "lr",
              "image_size", "seed", "device"]:
        print(f"   {k:<12s} {getattr(args, k)}")
    print(f"PyTorch  : {'已安装 ' + torch.__version__ if HAS_TORCH else '未安装（dry-run 仍然可用）'}")
    print("\n[OK] 数据管线自检通过。以上是实际扫描结果，尚未进行任何训练。")
    return 0


def print_model(args):
    if not HAS_TORCH:
        raise SystemExit("--print-model 需要 PyTorch。")
    model, note = build_model(args.model, len(CLASS_NAMES), args.pretrained,
                              args.freeze_backbone)
    total, trainable = count_params(model)
    print(f"model : {note}")
    n_params = 0
    print(f"{'layer':<40s}{'type':<22s}{'out shape':<20s}{'params':>10s}")
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 and name:
            p = sum(x.numel() for x in mod.parameters(recurse=False))
            if p == 0:      # 跳过 ReLU / Dropout / Pooling 这类没有参数的层
                continue
            n_params += p
            print(f"{name[:39]:<40s}{type(mod).__name__:<22s}{'-':<20s}{p:>10,}")
    print("-" * 92)
    print(f"{'TOTAL':<40s}{'':<22s}{'':<20s}{total:>10,}")
    print(f"trainable params: {trainable:,}")
    # 用 dummy 输入跑一次，确认输出类别数
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, args.image_size, args.image_size))
    print(f"dummy input [1,3,{args.image_size},{args.image_size}] -> output {tuple(out.shape)}")
    print("[提示] 把 TOTAL 填进报告 3.4 的参数量单元格。")
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
            raise SystemExit("--dry-run 需要 --data-root（PlantVillage 彩色图目录）")
        return dry_run(args)
    if args.eval_only:
        if args.ckpt is None:
            raise SystemExit("--eval-only 需要 --ckpt（训练好的 best.pt）")
        if args.plantdoc_root is None and args.data_root is None:
            raise SystemExit("--eval-only 需要 --plantdoc-root（跨域）或 --data-root（in-domain）")
        if not HAS_TORCH:
            raise SystemExit("评估也需要 PyTorch：pip install -r requirements.txt")
        return eval_only(args)
    if not HAS_TORCH:
        raise SystemExit("没有安装 PyTorch：pip install -r requirements.txt（或先用 --dry-run 检查数据）")
    if args.data_root is None:
        raise SystemExit("训练需要 --data-root（PlantVillage 彩色图目录）")
    return train(args)


def eval_only(args):
    """加载权重，只在 PlantDoc（或 PV test）上评估，不对模型做任何更新。"""
    device = (torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available())
              else torch.device("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # 评估时不需要再下载预训练权重：状态字典会整体覆盖
    model, note = build_model(ckpt.get("model", args.model), len(CLASS_NAMES),
                              pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print(f"[INFO] 已加载 {args.ckpt}（{note}），不做任何训练。")
    mapping = load_mapping(args.mapping)
    if args.plantdoc_root:
        # 跨域：PlantDoc 整份数据都只当测试集用
        items, _stats = build_index(args.plantdoc_root, "plantdoc", mapping,
                                    split_filter=args.plantdoc_split)
        eval_items, split_note = items, f"PlantDoc[{args.plantdoc_split}]"
        tag = "plantdoc"
    else:
        # 域内：必须用和训练时同一套 split，只取 test 部分。
        # 否则会把训练图片算进"测试准确率"，数字是虚高的。
        items, _stats = build_index(args.data_root, args.dataset_name, mapping,
                                    args.limit_per_class)
        r = [float(x) for x in args.split.split("/")]
        r = [x / sum(r) for x in r]
        _tr, _va, eval_items = stratified_split(items, tuple(r), args.seed)
        split_note = f"PlantVillage test split (split_seed={args.seed})"
        tag = "test"
    if not eval_items:
        raise SystemExit("评估集为空，请检查路径与类名映射。")
    print(f"[INFO] 评估集 = {split_note}，共 {len(eval_items)} 张")
    ds = ImageListDataset(eval_items, build_transforms(args.image_size, False)[1])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss()
    m = evaluate(model, dl, criterion, device, len(CLASS_NAMES), collect=True)
    print(f"[EVAL] acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} n={m['n']}")
    save_metrics(m, args.out_dir, tag)

    # 逐图预测（标签感知文件名：predictions_plantdoc.csv / predictions_test.csv）
    # 错例分析（Fig. 6）需要它，跨域尤其重要。
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
