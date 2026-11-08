#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py
===============
把 `runs/` 里的训练产物批量画成报告要用的图（不需要 PyTorch，只需要 matplotlib + Pillow）。

输入（train_baseline.py 的输出）
--------------------------------
    runs/<run_name>/log.csv                    每 epoch 的 loss / acc / macro-F1
    runs/<run_name>/metrics_test.json          PV 测试集指标（acc / macro_f1 / 混淆矩阵）
    runs/<run_name>/metrics_plantdoc.json      PlantDoc 跨域指标（E5 才有）
    runs/<run_name>/confusion_matrix_*.csv     3×3 混淆矩阵
    runs/<run_name>/predictions_test.csv       逐图预测（用来挑错例）

输出（--out-dir，默认 figures/）
--------------------------------
    fig3_curves_<run>.png        训练曲线：loss + 验证 acc / macro-F1
    fig4_cm_<run>_<tag>.png      混淆矩阵热图（tag = test / plantdoc）
    fig5_compare.png             各实验 Acc / macro-F1 对比（含 PlantDoc 跨域）
    fig6_errors_<run>.png        错例拼图（真实标签 → 预测标签）
    figures/index.md             本次生成了哪些图 + 可直接用的英文图注

用法
----
    # 全部自动发现
    python plot_results.py --runs runs --out-dir figures

    # 只画某一组
    python plot_results.py --runs runs --only E1 --out-dir figures

    # 错例图指定用哪次 run（默认取第一个有 predictions_test.csv 的）
    python plot_results.py --runs runs --errors-run E5_plantdoc_eval

说明
----
* 所有图里的数字都直接来自文件，脚本不写死任何数值。
* 缺文件就跳过并打印提示，不会报错中断。
* 图里不画平滑曲线、不裁剪 epoch，避免"看起来更好"的误导。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_ORDER = ["healthy", "early_blight", "late_blight"]
SHORT = {"healthy": "healthy", "early_blight": "early blight", "late_blight": "late blight"}


# --------------------------------------------------------------------------- #
# 读文件
# --------------------------------------------------------------------------- #
def read_csv_rows(p: Path):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_runs(runs_dir: Path, only: str | None):
    """只把真正含训练产物的目录当成 run，避免把图片夹/缓存目录也算进去。

    另外按约定跳过"实验性/作废"的目录：名字以 `_` 开头，或含 smoke / ignore /
    checkpoint 的（例如 `_smoke_test_ignore`），免得冒烟测试的数据混进对比图。
    """
    runs, skipped = [], []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if only and only.lower() not in d.name.lower():
            continue
        low = d.name.lower()
        if d.name.startswith("_") or any(k in low for k in ("smoke", "ignore", "tmp", "checkpoint")):
            skipped.append(d.name)
            continue
        has_output = (d / "log.csv").exists() or any(d.glob("metrics_*.json")) \
            or (d / "predictions_test.csv").exists()
        if not has_output:
            continue
        runs.append(d)
    if skipped:
        print(f"[INFO] 已跳过 {len(skipped)} 个非正式目录：{', '.join(skipped)}")
    return runs


# --------------------------------------------------------------------------- #
# 图 3：训练曲线
# --------------------------------------------------------------------------- #
def plot_curves(run_dir: Path, out_dir: Path, run_label: str):
    log = run_dir / "log.csv"
    if not log.exists():
        return None
    rows = read_csv_rows(log)
    if not rows:
        return None
    ep = [int(r["epoch"]) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    va_acc = [float(r["val_acc"]) for r in rows]
    va_f1 = [float(r["val_macro_f1"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax1.plot(ep, tr_loss, marker="o", ms=3, label="train loss")
    ax1.plot(ep, va_loss, marker="s", ms=3, label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend(frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(ep, va_acc, marker="o", ms=3, label="val accuracy")
    ax2.plot(ep, va_f1, marker="s", ms=3, label="val macro-F1")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Validation performance")
    ax2.legend(frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{run_label} — training curves "
                 f"(best val macro-F1 = {max(va_f1):.4f} at epoch "
                 f"{ep[int(np.argmax(va_f1))]})", fontsize=10)
    fig.tight_layout()
    out = out_dir / f"fig3_curves_{safe(run_label)}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    return out.name


# --------------------------------------------------------------------------- #
# 图 4：混淆矩阵热图
# --------------------------------------------------------------------------- #
def plot_cm(run_dir: Path, out_dir: Path, run_label: str):
    made = []
    for cm_file in sorted(run_dir.glob("confusion_matrix_*.csv")):
        tag = cm_file.stem.replace("confusion_matrix_", "")
        rows = read_csv_rows(cm_file)
        if not rows:
            continue
        labels = [k for k in rows[0].keys() if k != "true\\pred"]
        cm = np.array([[int(float(r[l])) for l in labels] for r in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([SHORT.get(l, l) for l in labels], rotation=20, ha="right")
        ax.set_yticklabels([SHORT.get(l, l) for l in labels])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = int(cm[i, j])
                share = v / total if total else 0
                ax.text(j, i, f"{v}\n({share:.0%})", ha="center", va="center",
                        fontsize=8, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
        ax.set_title(f"{run_label} — {tag} (n = {int(total)})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = out_dir / f"fig4_cm_{safe(run_label)}_{tag}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out}")
        made.append(out.name)
    return made


# --------------------------------------------------------------------------- #
# 图 5：各实验对比
# --------------------------------------------------------------------------- #
def collect_metrics(runs):
    """返回 [(label, {test:..., plantdoc:...})]"""
    out = []
    for d in runs:
        item = {"label": d.name, "test": None, "plantdoc": None}
        for tag in ("test", "plantdoc"):
            p = d / f"metrics_{tag}.json"
            if p.exists():
                item[tag] = read_json(p)
        if item["test"] or item["plantdoc"]:
            out.append(item)
    return out


def plot_compare(runs, out_dir: Path):
    data = collect_metrics(runs)
    if len(data) < 2:
        print("[SKIP] 少于 2 组 metrics，跳过对比图（先跑完 E1 和 E3 再来）")
        return None

    labels = [d["label"] for d in data]
    x = np.arange(len(labels))
    w = 0.36

    # 左右两个子图：左=accuracy，右=macro-F1；每个子图里 PV-test 与 PlantDoc 并排。
    # 之所以不用"4 个系列一张图"，是因为 4 组柱子会挤在一起、图例也容易压住数字。
    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 1.7 * len(labels)), 4.2), sharey=True)
    panels = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]
    for pi, (ax, (key, title)) in enumerate(zip(axes, panels)):
        pv = [(d["test"] or {}).get(key) for d in data]
        pd_ = [(d["plantdoc"] or {}).get(key) for d in data]
        for offset, vals, name, color in ((-w / 2, pv, "PlantVillage test", "#3B6EA5"),
                                          (w / 2, pd_, "PlantDoc", "#B23A48")):
            pts = [(xi + offset, v) for xi, v in zip(x, vals) if v is not None]
            if not pts:
                continue
            ax.bar([p[0] for p in pts], [p[1] for p in pts], width=w, label=name,
                   color=color, edgecolor="black", linewidth=0.5)
            for xi, v in pts:
                ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.06)
        ax.spines[["top", "right"]].set_visible(False)
        if pi == 0:      # 图例放在左图下方，避免压住任何柱子
            ax.legend(frameon=False, fontsize=8, loc="upper center",
                      bbox_to_anchor=(0.5, -0.26), ncol=2)
    axes[0].set_ylabel("score")

    fig.suptitle("In-domain vs. cross-domain performance "
                 "(values read from metrics_*.json; absent bars = not run yet)", fontsize=10)
    fig.tight_layout()
    out = out_dir / "fig5_compare.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    return out.name, len(data)


# --------------------------------------------------------------------------- #
# 图 6：错例拼图
# --------------------------------------------------------------------------- #
def plot_errors(run_dir: Path, out_dir: Path, run_label: str, n=6, min_prob=0.0,
                tag: str = "plantdoc"):
    """错例拼图。默认优先用跨域（PlantDoc）的预测，因为它才是错误分析的重点。"""
    pred = run_dir / f"predictions_{tag}.csv"
    if not pred.exists():
        alt = run_dir / "predictions_test.csv"
        if tag != "test" and alt.exists():
            print(f"[INFO] {run_label}: 没有 predictions_{tag}.csv，退回用 test 集预测")
            pred, tag = alt, "test"
        else:
            return None
    rows = read_csv_rows(pred)
    wrong = [r for r in rows if str(r.get("correct", "0")) in ("0", "False", "false")]
    if not wrong:
        print(f"[INFO] {run_label}: {tag} 没有错例（或不含 correct 列），跳过错例图")
        return None

    def maxp(r):
        try:
            return max(float(r["p_healthy"]), float(r["p_early"]), float(r["p_late"]))
        except (KeyError, ValueError):
            return 0.0

    wrong.sort(key=maxp, reverse=True)
    picked = [r for r in wrong if maxp(r) >= min_prob][:n]
    if not picked:
        return None

    from PIL import Image
    cols = 3
    rows_n = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 3.05 * rows_n), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, r in enumerate(picked):
        ax = axes[i // cols][i % cols]
        p = Path(r["path"])
        if p.exists():
            try:
                ax.imshow(Image.open(p).convert("RGB"))
            except Exception:  # noqa: BLE001
                ax.text(0.5, 0.5, "cannot open", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "image not found\n" + p.name, ha="center", va="center",
                    fontsize=8)
        # 标签写在图内左上角（不放在标题位置，否则图片宽高比不一致时会和上一行重叠）
        ax.text(0.03, 0.97,
                f"true: {SHORT.get(r['true'], r['true'])}\n"
                f"pred: {SHORT.get(r['pred'], r['pred'])}  (p={maxp(r):.2f})",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="white",
                bbox=dict(facecolor="black", alpha=0.6, pad=2.5, edgecolor="none"))
    fig.suptitle(f"{run_label} — misclassified test images (top-{len(picked)} by confidence)",
                 fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02,
                        wspace=0.05, hspace=0.06)
    out = out_dir / f"fig6_errors_{safe(run_label)}_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    return out.name, len(picked), len(rows), len(wrong)


def safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


# --------------------------------------------------------------------------- #
# 图注清单
# --------------------------------------------------------------------------- #
CAPTIONS = {
    "fig3": "Fig. 3 Training curves. Left: training and validation cross-entropy loss per epoch. "
            "Right: validation accuracy and validation macro-F1 per epoch. The checkpoint with "
            "the highest validation macro-F1 is kept for evaluation.",
    "fig4": "Fig. 4 Confusion matrices on the three-class tomato task; cell values are image "
            "counts with the row-normalised share in brackets.",
    "fig5": "Fig. 5 In-domain (PlantVillage test) versus cross-domain (PlantDoc) accuracy and "
            "macro-F1 for every configuration. All values are read from the corresponding "
            "metrics_*.json files.",
    "fig6": "Fig. 6 Misclassified test images. Each panel shows the ground-truth label, the "
            "predicted label, and the predicted probability, illustrating the failure modes "
            "discussed in the error analysis.",
}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="把 runs/ 里的结果画成报告用的图")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--only", default=None, help="只处理名字包含该字符串的 run")
    ap.add_argument("--errors-run", default=None,
                    help="错例图用哪个 run（默认第一个有 predictions_test.csv 的）")
    ap.add_argument("--errors-n", type=int, default=6)
    ap.add_argument("--errors-tag", default="plantdoc", choices=["plantdoc", "test"],
                    help="错例图用哪一组预测，默认跨域 PlantDoc")
    args = ap.parse_args(argv)

    if not args.runs.exists():
        print(f"[ERROR] 找不到 {args.runs}，先跑训练。", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = find_runs(args.runs, args.only)
    if not runs:
        print(f"[ERROR] {args.runs} 下没有 run 目录。", file=sys.stderr)
        return 2
    print(f"[INFO] 发现 {len(runs)} 个 run：" + ", ".join(d.name for d in runs))

    made = []
    for d in runs:
        f = plot_curves(d, args.out_dir, d.name)
        if f:
            made.append(("fig3", f))
        for f in plot_cm(d, args.out_dir, d.name):
            made.append(("fig4", f))

    cmp_result = plot_compare(runs, args.out_dir)
    if cmp_result:
        made.append(("fig5", cmp_result[0]))

    # 错例图
    err_run = None
    if args.errors_run:
        err_run = next((d for d in runs if args.errors_run.lower() in d.name.lower()), None)
    if err_run is None:
        err_run = next((d for d in runs if (d / "predictions_test.csv").exists()), None)
    if err_run is not None:
        r = plot_errors(err_run, args.out_dir, err_run.name, n=args.errors_n, tag=args.errors_tag)
        if r:
            made.append(("fig6", r[0]))

    # index.md
    with open(args.out_dir / "index.md", "w", encoding="utf-8") as f:
        f.write("# 生成的图与图注\n\n")
        f.write("> 图注可直接粘进报告；`Fig. N` 的编号按你们 Word 里的顺序改。\n\n")
        seen = set()
        for key, name in made:
            if key in seen:
                continue
            seen.add(key)
            f.write(f"## `{name}`\n\n{CAPTIONS[key]}\n\n")
        f.write("\n---\n\n## 本次读到的数值（核对用）\n\n")
        for item in collect_metrics(runs):
            t = item["test"] or {}
            p = item["plantdoc"] or {}
            acc_t = t.get("accuracy", t.get("acc"))
            f1_t = t.get("macro_f1")
            acc_p = p.get("accuracy", p.get("acc"))
            f1_p = p.get("macro_f1")
            line = f"- **{item['label']}**: PV-test acc={acc_t}, macro-F1={f1_t}"
            if acc_p is not None:
                line += f"; PlantDoc acc={acc_p}, macro-F1={f1_p}"
                if acc_t is not None and acc_p is not None:
                    line += f"; Δacc={acc_t - acc_p:+.4f}"
            f.write(line + "\n")
    print(f"[OK] {args.out_dir / 'index.md'}")
    print(f"\n共生成 {len(made)} 个图文件。下一步：打开 figures/index.md 拿图注，"
          f"把数字抄进报告的 Table 9。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
