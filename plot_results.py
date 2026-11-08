#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py
===============
Turn the artefacts in `runs/` into the figures used in the report (matplotlib + Pillow only).

Input (produced by train_baseline.py)
--------------------------------
    runs/<run_name>/log.csv                    loss / acc / macro-F1 per epoch
    runs/<run_name>/metrics_test.json          PlantVillage test metrics
    runs/<run_name>/metrics_plantdoc.json      PlantDoc cross-domain metrics
    runs/<run_name>/confusion_matrix_*.csv     3 x 3 confusion matrix
    runs/<run_name>/predictions_test.csv       per-image predictions, used to pick failures

Output (--out-dir, default figures/)
--------------------------------
    fig3_curves_<run>.png        training curves: loss and validation scores
    fig4_cm_<run>_<tag>.png      confusion matrix heat map (tag = test / plantdoc)
    fig5_compare.png             accuracy and macro-F1 per configuration, both domains
    fig6_errors_<run>.png        misclassified images (true label to predicted label)
    figures/index.md             which figures were produced, with ready-to-use captions

Usage
----
    # discover every run
    python plot_results.py --runs runs --out-dir figures

    # a single run
    python plot_results.py --runs runs --only E1 --out-dir figures

    # which run to use for the error panel (default: the first with predictions_test.csv)
    python plot_results.py --runs runs --errors-run E5_plantdoc_eval

Notes
----
* Every number in the figures is read from a file; nothing is hard-coded.
* A missing file is skipped with a message instead of aborting the script.
* No smoothing and no epoch trimming, so the curves cannot look better than they are.
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
# File readers
# --------------------------------------------------------------------------- #
def read_csv_rows(p: Path):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_runs(runs_dir: Path, only: str | None):
    """Treat only folders that hold training output as runs, not image or cache folders.

    Folders that start with `_`, or that contain smoke / ignore / checkpoint,
    are skipped so that smoke-test output cannot leak into the comparison figure."""
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
        print(f"[INFO] skipped {len(skipped)} non-official folders: {', '.join(skipped)}")
    return runs


# --------------------------------------------------------------------------- #
# Figure 3: training curves
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
# Figure 4: confusion matrix
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
# Figure 5: comparison across configurations
# --------------------------------------------------------------------------- #
def collect_metrics(runs):
    """Return [(label, {test: ..., plantdoc: ...})]."""
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
        print("[SKIP] fewer than two runs have metrics; the comparison figure needs at least E1 and E3")
        return None

    labels = [d["label"] for d in data]
    x = np.arange(len(labels))
    w = 0.36

    # Two panels: accuracy on the left, macro-F1 on the right; both domains side by side.
    # Four series in one panel would crowd the bars and hide the numbers behind the legend.
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
        if pi == 0:      # legend below the left panel so that no bar is covered
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
# Figure 6: error panel
# --------------------------------------------------------------------------- #
def plot_errors(run_dir: Path, out_dir: Path, run_label: str, n=6, min_prob=0.0,
                tag: str = "plantdoc"):
    """Error panel; by default it uses the cross-domain predictions, which matter most."""
    pred = run_dir / f"predictions_{tag}.csv"
    if not pred.exists():
        alt = run_dir / "predictions_test.csv"
        if tag != "test" and alt.exists():
            print(f"[INFO] {run_label}: no predictions_{tag}.csv, falling back to the test split")
            pred, tag = alt, "test"
        else:
            return None
    rows = read_csv_rows(pred)
    wrong = [r for r in rows if str(r.get("correct", "0")) in ("0", "False", "false")]
    if not wrong:
        print(f"[INFO] {run_label}: no errors in {tag}, skipping the error panel")
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
        # Labels sit inside the image, top left; as titles they would collide with the row above
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
# Caption list
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
    ap = argparse.ArgumentParser(description="Turn the results in runs/ into report figures")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--only", default=None, help="only process runs whose name contains this")
    ap.add_argument("--errors-run", default=None,
                    help="run used for the error panel (default: first with predictions_test.csv)")
    ap.add_argument("--errors-n", type=int, default=6)
    ap.add_argument("--errors-tag", default="plantdoc", choices=["plantdoc", "test"],
                    help="which predictions the error panel uses (default: cross-domain PlantDoc)")
    args = ap.parse_args(argv)

    if not args.runs.exists():
        print(f"[ERROR] {args.runs} does not exist; run the training first.", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = find_runs(args.runs, args.only)
    if not runs:
        print(f"[ERROR] no run folders inside {args.runs}.", file=sys.stderr)
        return 2
    print(f"[INFO] found {len(runs)} runs: " + ", ".join(d.name for d in runs))

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

    # Error panel
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
        f.write("# Generated figures and captions\n\n")
        f.write("> Captions can be pasted into the report; adjust `Fig. N` to your numbering.\n\n")
        seen = set()
        for key, name in made:
            if key in seen:
                continue
            seen.add(key)
            f.write(f"## `{name}`\n\n{CAPTIONS[key]}\n\n")
        f.write("\n---\n\n## Values read from the runs (for checking)\n\n")
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
    print(f"\nWrote {len(made)} figure files. Next: open figures/index.md for the captions "
          f"and copy the numbers into Table 5 of the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
