"""
Evaluasi Lengkap Model YOLOv8 (ONNX) - Subbab 3.5 Proposal Tugas Akhir

Menjalankan model yang SAMA dengan aplikasi (models/wound_yolov8n.onnx) pada
split train / valid / test, lalu menghitung:

  A. Evaluasi tingkat citra  (image-level)  : confusion matrix K x (K+1), TP/FP/FN/TN,
                                              Precision, Recall, Specificity, F1, Kappa, MCC, CI 95%.
  B. Evaluasi tingkat deteksi (box-level)   : Precision, Recall, AP@0.5, AP@0.75, AP@0.5:0.95, mAP.
  C. Analisis pendukung                     : sweep threshold, kecepatan inferensi,
                                              kesenjangan train-valid-test, analisis kesalahan.

Keluaran: reports/metrics.json + gambar PNG di reports/.
Setelah itu jalankan:  python scripts/generate_report.py

Pemakaian:
    python scripts/evaluate_full.py
    python scripts/evaluate_full.py --conf 0.35 --iou 0.45 --splits valid test
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(BASE_DIR, "app")
for p in (APP_DIR,):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.detector import WoundONNXDetector  # noqa: E402
from core import metrics as M  # noqa: E402

CLASS_NAMES = ["Abrasions", "Bruise", "Burn", "Cut", "No Abnormality"]
CLASS_ID = ["Luka lecet", "Luka memar", "Luka bakar", "Luka sayat", "Kulit normal"]
CLASS_ABBR = ["Abras", "Bruise", "Burn", "Cut", "Normal"]
SPLIT_TITLES = {"train": "Training (Data Latih)", "valid": "Validasi", "test": "Testing (Data Uji)"}
SPLIT_KEY = {"train": "training", "valid": "validasi", "test": "testing"}
K = len(CLASS_NAMES)
PALETTE = {"primary": "#9E3B3B", "amber": "#D99B26", "slate": "#526B7A", "green": "#3B7A57", "ink": "#2B1E1E"}


# ------------------------------------------------------------------------------
# Inferensi
# ------------------------------------------------------------------------------
def read_gt(label_path, w, h):
    """Baca label YOLO -> boxes xyxy piksel & kelas."""
    boxes, cls = [], []
    if os.path.exists(label_path):
        with open(label_path, "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                xc, yc, bw, bh = [float(v) for v in parts[1:5]]
                boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
                cls.append(c)
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(cls, dtype=int)


def image_level_truth(gt_boxes, gt_cls):
    """Kelas citra = kelas dominan (jumlah box terbanyak; seri -> total luas terbesar)."""
    if len(gt_cls) == 0:
        return None
    cnt = Counter(gt_cls.tolist())
    top = max(cnt.values())
    cands = [c for c, n in cnt.items() if n == top]
    if len(cands) == 1:
        return cands[0]
    areas = {c: float(sum((b[2] - b[0]) * (b[3] - b[1]) for b, k in zip(gt_boxes, gt_cls) if k == c)) for c in cands}
    return max(areas, key=areas.get)


def run_split(det, split, iou_nms, max_det=300, limit=None):
    img_dir = os.path.join(BASE_DIR, "dataset", split, "images")
    lbl_dir = os.path.join(BASE_DIR, "dataset", split, "labels")
    files = sorted(os.listdir(img_dir))
    if limit:
        files = files[:limit]
    session = det.session
    in_name = session.get_inputs()[0].name

    records = []
    t_pre, t_inf, t_post = [], [], []
    t0 = time.time()
    for n, f in enumerate(files, 1):
        path = os.path.join(img_dir, f)
        img = np.array(Image.open(path).convert("RGB"))
        h0, w0 = img.shape[:2]
        gt_boxes, gt_cls = read_gt(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"), w0, h0)

        a = time.perf_counter()
        blob, scale, pad_x, pad_y = det._preprocess(img)
        b = time.perf_counter()
        out = session.run(None, {in_name: blob})[0][0].T          # (8400, 4+K)
        c = time.perf_counter()

        raw_boxes, scores = out[:, :4], out[:, 4:]
        max_s = scores.max(axis=1)
        max_c = scores.argmax(axis=1)
        top_i = int(np.argmax(max_s))

        # --- Deteksi multi-label per kelas (conf 0.001) + NMS per kelas ---
        cx, cy, bw, bh = raw_boxes[:, 0], raw_boxes[:, 1], raw_boxes[:, 2], raw_boxes[:, 3]
        xyxy = np.stack([(cx - bw / 2 - pad_x) / scale, (cy - bh / 2 - pad_y) / scale,
                         (cx + bw / 2 - pad_x) / scale, (cy + bh / 2 - pad_y) / scale], axis=1)
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w0)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h0)

        d_boxes, d_scores, d_cls = [], [], []
        for k in range(K):
            m = scores[:, k] > 0.001
            if not m.any():
                continue
            idx = np.where(m)[0]
            if len(idx) > 1000:
                idx = idx[np.argsort(-scores[idx, k])[:1000]]
            keep = M.nms_numpy(xyxy[idx], scores[idx, k], iou_nms)
            for j in keep:
                d_boxes.append(xyxy[idx[j]])
                d_scores.append(scores[idx[j], k])
                d_cls.append(k)
        if d_scores:
            order = np.argsort(-np.array(d_scores))[:max_det]
            d_boxes = np.array(d_boxes)[order]
            d_scores = np.array(d_scores)[order]
            d_cls = np.array(d_cls)[order]
        else:
            d_boxes, d_scores, d_cls = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        d = time.perf_counter()

        t_pre.append((b - a) * 1000)
        t_inf.append((c - b) * 1000)
        t_post.append((d - c) * 1000)

        records.append({
            "file": f,
            "true_image": image_level_truth(gt_boxes, gt_cls),
            "top_class": int(max_c[top_i]),
            "top_score": float(max_s[top_i]),
            "gt": {"boxes": gt_boxes, "classes": gt_cls},
            "pred": {"boxes": d_boxes, "scores": d_scores, "classes": d_cls},
        })
        if n % 200 == 0 or n == len(files):
            print(f"   [{split}] {n}/{len(files)} citra ({time.time() - t0:.0f} dtk)", flush=True)

    timing = {"pre_ms": float(np.mean(t_pre)), "infer_ms": float(np.mean(t_inf)), "post_ms": float(np.mean(t_post)),
              "total_ms": float(np.mean(np.array(t_pre) + np.array(t_inf) + np.array(t_post))),
              "infer_ms_median": float(np.median(t_inf)), "infer_ms_p95": float(np.percentile(t_inf, 95))}
    timing["fps"] = 1000.0 / timing["total_ms"] if timing["total_ms"] else 0.0
    return records, timing


# ------------------------------------------------------------------------------
# Metrik
# ------------------------------------------------------------------------------
def image_predictions(records, conf):
    """Prediksi citra = kelas deteksi dengan skor tertinggi bila >= conf, jika tidak -1."""
    yt, yp, keep = [], [], []
    for i, r in enumerate(records):
        if r["true_image"] is None:
            continue
        yt.append(r["true_image"])
        yp.append(r["top_class"] if r["top_score"] >= conf else -1)
        keep.append(i)
    return np.array(yt), np.array(yp), keep


def evaluate_split(records, conf, n_boot):
    yt, yp, keep = image_predictions(records, conf)
    cm = M.build_confusion_matrix(yt, yp, K)
    pc = M.per_class_metrics(cm, CLASS_NAMES)
    ov = M.overall_metrics(cm, CLASS_NAMES)
    ci = M.bootstrap_ci(yt, yp, K, n_boot=n_boot)
    for c in CLASS_NAMES:
        pc[c]["recall_ci95"] = M.wilson_interval(pc[c]["tp"], pc[c]["support"])
        pc[c]["precision_ci95"] = M.wilson_interval(pc[c]["tp"], pc[c]["tp"] + pc[c]["fp"])
    ov["ci95"] = ci

    det = M.detection_ap([r["pred"] for r in records], [r["gt"] for r in records], K, conf_for_pr=conf)
    curves = det.pop("pr_curves")
    det_pc = {CLASS_NAMES[c]: v for c, v in det.pop("per_class").items()}

    mis = []
    for pos, i in enumerate(keep):
        if yt[pos] != yp[pos]:
            r = records[i]
            mis.append({"file": r["file"], "true": CLASS_NAMES[yt[pos]],
                        "pred": CLASS_NAMES[yp[pos]] if yp[pos] >= 0 else M.NONE_LABEL,
                        "score": r["top_score"], "top_class": CLASS_NAMES[r["top_class"]]})
    return {
        "total_images": int(len(records)),
        "images_without_label": int(len(records) - len(keep)),
        "confusion_matrix": {"labels": CLASS_NAMES, "labels_id": CLASS_ID, "labels_abbr": CLASS_ABBR,
                             "columns": CLASS_NAMES + [M.NONE_LABEL], "matrix": cm.tolist()},
        "metrics_overall": ov,
        "metrics_per_class": {c: {**pc[c], "label_id": CLASS_ID[i]} for i, c in enumerate(CLASS_NAMES)},
        "top_confusions": M.top_confusions(cm, CLASS_NAMES),
        "detection": {"overall": det, "per_class": det_pc},
        "misclassified": mis,
    }, curves


def threshold_sweep(records, thresholds):
    rows = []
    for t in thresholds:
        yt, yp, _ = image_predictions(records, t)
        cm = M.build_confusion_matrix(yt, yp, K)
        ov = M.overall_metrics(cm, CLASS_NAMES)
        rows.append({"threshold": float(round(t, 2)), "accuracy": ov["accuracy"], "f1_macro": ov["f1_macro"],
                     "precision_macro": ov["precision_macro"], "recall_macro": ov["recall_macro"],
                     "undetected_rate": ov["undetected"] / max(ov["total"], 1)})
    return rows


def read_training_curve(run_dir):
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({k.strip(): float(v) for k, v in r.items() if v.strip() != ""})
    return rows


# ------------------------------------------------------------------------------
# Visualisasi
# ------------------------------------------------------------------------------
def make_plots(out_dir, results, curves_test, sweeps, train_rows, test_records, timing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    cols = CLASS_ABBR + ["Tdk Terdeteksi"]
    saved = {}

    def draw_cm(ax, mat, title, normalize):
        mat = np.array(mat, dtype=float)
        shown = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1) if normalize else mat
        im = ax.imshow(shown, cmap="Reds" if normalize else "Blues", vmin=0, vmax=1 if normalize else None)
        ax.set_xticks(range(K + 1)); ax.set_yticks(range(K))
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8, fontweight="bold")
        ax.set_yticklabels(CLASS_ABBR, fontsize=8, fontweight="bold")
        thr = shown.max() / 2 if shown.max() else 0
        for i in range(K):
            for j in range(K + 1):
                v = shown[i, j]
                txt = f"{v * 100:.0f}%" if normalize else f"{int(v)}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if v > thr else PALETTE["ink"], fontweight="bold" if i == j else "normal")
        ax.axvline(K - 0.5, color="#888", lw=1, ls="--")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Prediksi Model"); ax.set_ylabel("Kondisi Aktual")
        return im

    # 1. Confusion matrix per split (jumlah & ternormalisasi) + gabungan
    for sp, res in results.items():
        mat = res["confusion_matrix"]["matrix"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
        draw_cm(axes[0], mat, f"{SPLIT_TITLES[sp]} - Jumlah Citra (N={res['metrics_overall']['total']})", False)
        draw_cm(axes[1], mat, f"{SPLIT_TITLES[sp]} - Persentase per Kelas Aktual (Recall)", True)
        plt.tight_layout()
        p = os.path.join(out_dir, f"confusion_matrix_{SPLIT_KEY[sp]}.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved[f"cm_{sp}"] = p

    fig, axes = plt.subplots(1, len(results), figsize=(6.4 * len(results), 5.2), dpi=200)
    axes = np.atleast_1d(axes)
    for ax, (sp, res) in zip(axes, results.items()):
        draw_cm(ax, res["confusion_matrix"]["matrix"], f"{SPLIT_TITLES[sp]} (N={res['metrics_overall']['total']})", False)
    fig.suptitle("Matriks Konfusi Tingkat Citra (kolom terakhir: citra tidak terdeteksi)", fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "confusion_matrix.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    saved["cm_combined"] = p

    # 2. Metrik per kelas (test)
    if "test" in results:
        pcm = results["test"]["metrics_per_class"]
        x = np.arange(K); w = 0.26
        fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=200)
        for off, key, color, name in ((-w, "precision", PALETTE["primary"], "Precision"),
                                      (0, "recall", PALETTE["amber"], "Recall"),
                                      (w, "f1_score", PALETTE["slate"], "F1-Score")):
            vals = [pcm[c][key] * 100 for c in CLASS_NAMES]
            bars = ax.bar(x + off, vals, w, label=name, color=color)
            for b_, v in zip(bars, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES); ax.set_ylim(0, 112); ax.set_ylabel("%")
        ax.set_title("Metrik per Kelas pada Data Uji (Tingkat Citra)", fontweight="bold")
        ax.legend(ncol=3, loc="upper center", frameon=False)
        plt.tight_layout(); p = os.path.join(out_dir, "per_class_metrics.png")
        fig.savefig(p); plt.close(fig); saved["per_class"] = p

    # 3. Kurva PR (test, IoU 0.5)
    if curves_test:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        colors = [PALETTE["primary"], PALETTE["amber"], "#C2703D", PALETTE["slate"], PALETTE["green"]]
        for c, cv in curves_test.items():
            ap = results["test"]["detection"]["per_class"][CLASS_NAMES[c]]["ap50"]
            ax.plot(cv["recall"], cv["precision"], color=colors[c], lw=2, label=f"{CLASS_NAMES[c]} (AP50={ap:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.02)
        m50 = results["test"]["detection"]["overall"]["map50"]
        ax.set_title(f"Kurva Precision-Recall @IoU 0.5 (mAP50 = {m50:.3f})", fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        plt.tight_layout(); p = os.path.join(out_dir, "pr_curve_test.png")
        fig.savefig(p); plt.close(fig); saved["pr"] = p

    # 4. Sweep threshold
    fig, axes = plt.subplots(1, len(sweeps), figsize=(6 * len(sweeps), 4.2), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (sp, rows) in zip(axes, sweeps.items()):
        t = [r["threshold"] for r in rows]
        ax.plot(t, [r["accuracy"] * 100 for r in rows], color=PALETTE["primary"], lw=2, label="Akurasi")
        ax.plot(t, [r["f1_macro"] * 100 for r in rows], color=PALETTE["slate"], lw=2, label="Macro-F1")
        ax.plot(t, [r["undetected_rate"] * 100 for r in rows], color=PALETTE["amber"], lw=2, ls="--", label="% Tidak Terdeteksi")
        ax.axvline(0.35, color="#444", ls=":", lw=1.2); ax.text(0.36, 5, "threshold\naplikasi 0.35", fontsize=7)
        ax.set_title(f"Sweep Threshold - {SPLIT_TITLES[sp]}", fontweight="bold"); ax.set_xlabel("Confidence threshold")
    axes[0].set_ylabel("%"); axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); p = os.path.join(out_dir, "threshold_sweep.png"); fig.savefig(p); plt.close(fig)
    saved["sweep"] = p

    # 5. Kesenjangan generalisasi
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    sps = list(results.keys()); x = np.arange(len(sps)); w = 0.2
    series = [("Akurasi Citra", [results[s]["metrics_overall"]["accuracy"] for s in sps], PALETTE["primary"]),
              ("Macro-F1 Citra", [results[s]["metrics_overall"]["f1_macro"] for s in sps], PALETTE["slate"]),
              ("mAP@0.5", [results[s]["detection"]["overall"]["map50"] for s in sps], PALETTE["amber"]),
              ("mAP@0.5:0.95", [results[s]["detection"]["overall"]["map50_95"] for s in sps], PALETTE["green"])]
    for i, (name, vals, color) in enumerate(series):
        bars = ax.bar(x + (i - 1.5) * w, [v * 100 for v in vals], w, label=name, color=color)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v * 100 + 1, f"{v * 100:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([SPLIT_TITLES[s] for s in sps], fontsize=8); ax.set_ylim(0, 115); ax.set_ylabel("%")
    ax.set_title("Perbandingan Metrik Antar Split (Analisis Overfitting)", fontweight="bold")
    ax.legend(ncol=4, frameon=False, fontsize=7, loc="upper center")
    plt.tight_layout(); p = os.path.join(out_dir, "generalization_gap.png"); fig.savefig(p); plt.close(fig)
    saved["gap"] = p

    # 6. Kurva training
    if train_rows:
        ep = [r["epoch"] for r in train_rows]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=200)
        axes[0].plot(ep, [r["train/box_loss"] for r in train_rows], label="train box", color=PALETTE["primary"])
        axes[0].plot(ep, [r["val/box_loss"] for r in train_rows], label="val box", color=PALETTE["primary"], ls="--")
        axes[0].plot(ep, [r["train/cls_loss"] for r in train_rows], label="train cls", color=PALETTE["slate"])
        axes[0].plot(ep, [r["val/cls_loss"] for r in train_rows], label="val cls", color=PALETTE["slate"], ls="--")
        axes[0].set_title("Loss per Epoch", fontweight="bold"); axes[0].legend(frameon=False, fontsize=8)
        axes[1].plot(ep, [r["metrics/mAP50(B)"] for r in train_rows], color=PALETTE["amber"], label="mAP@0.5")
        axes[1].plot(ep, [r["metrics/mAP50-95(B)"] for r in train_rows], color=PALETTE["green"], label="mAP@0.5:0.95")
        axes[1].set_title("mAP Validasi per Epoch", fontweight="bold"); axes[1].legend(frameon=False, fontsize=8)
        axes[2].plot(ep, [r["metrics/precision(B)"] for r in train_rows], color=PALETTE["primary"], label="Precision")
        axes[2].plot(ep, [r["metrics/recall(B)"] for r in train_rows], color=PALETTE["slate"], label="Recall")
        axes[2].set_title("Precision & Recall Validasi per Epoch", fontweight="bold"); axes[2].legend(frameon=False, fontsize=8)
        for a in axes: a.set_xlabel("Epoch")
        plt.tight_layout(); p = os.path.join(out_dir, "training_curves.png"); fig.savefig(p); plt.close(fig)
        saved["training"] = p

    # 7. Contoh citra salah klasifikasi (test)
    mis = results.get("test", {}).get("misclassified", [])
    if mis:
        show = mis[:12]
        n = len(show); ncol = 4; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.3 * nrow), dpi=150)
        axes = np.atleast_1d(axes).flatten()
        for ax in axes: ax.axis("off")
        for ax, m in zip(axes, show):
            img = Image.open(os.path.join(BASE_DIR, "dataset", "test", "images", m["file"])).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"Aktual: {m['true']}\nPrediksi: {m['pred']} ({m['score']:.2f})", fontsize=7.5,
                         color=PALETTE["primary"])
        fig.suptitle(f"Citra Uji yang Salah Diklasifikasi ({len(mis)} citra, ditampilkan {n})", fontweight="bold")
        plt.tight_layout(); p = os.path.join(out_dir, "misclassified_test.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved["misclassified"] = p
    return saved


# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluasi lengkap model luka kulit (ONNX)")
    ap.add_argument("--model", default=os.path.join(BASE_DIR, "models", "wound_yolov8n.onnx"))
    ap.add_argument("--conf", type=float, default=0.35, help="Threshold evaluasi tingkat citra (default = aplikasi)")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU NMS (default = aplikasi)")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="Batasi jumlah citra per split (uji cepat)")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "reports"))
    ap.add_argument("--train-run", default=os.path.join(BASE_DIR, "runs", "detect", "runs", "train", "wound_yolov8-4"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("=" * 70)
    print(" EVALUASI LENGKAP MODEL YOLOv8 (ONNX) - Subbab 3.5")
    print(f" Model: {args.model}\n Threshold citra: {args.conf} | IoU NMS: {args.iou}")
    print("=" * 70)

    det = WoundONNXDetector(model_path=args.model)
    if det.session is None:
        sys.exit("[ERROR] Model ONNX tidak dapat dimuat.")

    results, curves, sweeps, timings, test_records = {}, {}, {}, {}, None
    for sp in args.splits:
        print(f"\n>> Inferensi split '{sp}' ...", flush=True)
        recs, tm = run_split(det, sp, args.iou, limit=args.limit)
        timings[sp] = tm
        print(f"   Menghitung metrik '{sp}' ...", flush=True)
        results[sp], cv = evaluate_split(recs, args.conf, args.bootstrap)
        if sp == "test":
            curves, test_records = cv, recs
        if sp in ("valid", "test"):
            sweeps[sp] = threshold_sweep(recs, np.arange(0.05, 0.951, 0.05))
        ov = results[sp]["metrics_overall"]; dt = results[sp]["detection"]["overall"]
        print(f"   [{sp}] N={ov['total']} Acc={ov['accuracy']:.4f} MacroF1={ov['f1_macro']:.4f} "
              f"mAP50={dt['map50']:.4f} mAP50-95={dt['map50_95']:.4f}")

    # Metadata dataset
    removed = {}
    rp = os.path.join(BASE_DIR, "dataset_removed", "removed_report.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as fh:
            removed = {k: len(v) for k, v in json.load(fh).items()}
    class_counts = {}
    for sp in ("train", "valid", "test"):
        d = os.path.join(BASE_DIR, "dataset", sp, "labels")
        cnt = Counter()
        if os.path.isdir(d):
            for f in os.listdir(d):
                _, c = read_gt(os.path.join(d, f), 1, 1)
                t = image_level_truth(np.zeros((len(c), 4)), c)
                if t is not None:
                    cnt[CLASS_NAMES[t]] += 1
        class_counts[sp] = dict(cnt)

    train_rows = read_training_curve(args.train_run)
    saved = make_plots(args.out, results, curves, sweeps, train_rows, test_records, timings)

    # Waktu inferensi rujukan = split test (atau split terakhir)
    ref = timings.get("test") or list(timings.values())[-1]
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": os.path.relpath(args.model, BASE_DIR),
            "conf_threshold": args.conf, "iou_nms": args.iou,
            "image_truth_rule": "kelas dominan (jumlah box terbanyak) pada label citra",
            "image_pred_rule": "kelas box skor tertinggi bila skor >= threshold, selain itu 'Tidak Terdeteksi'",
            "device": platform.processor() or platform.machine(),
            "platform": f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
            "training_run": os.path.relpath(args.train_run, BASE_DIR) if train_rows else None,
        },
        "dataset": {
            "class_names": CLASS_NAMES, "class_names_id": CLASS_ID,
            "splits": {sp: results[sp]["total_images"] for sp in results},
            "image_class_counts": class_counts,
            "removed_for_leakage": removed,
            "note_train": "Split train berisi citra hasil augmentasi Roboflow (±3x citra asli).",
        },
        "timing": {"reference_split": "test" if "test" in timings else list(timings)[-1], **ref, "per_split": timings},
        "evaluations": {SPLIT_KEY[sp]: {"name": f"Matrix Evaluasi ({SPLIT_TITLES[sp].split(' ')[0]})",
                                        "split_name": SPLIT_TITLES[sp], **res} for sp, res in results.items()},
        "threshold_sweep": sweeps,
        "training_curve_last_epoch": train_rows[-1] if train_rows else None,
        "figures": {k: os.path.relpath(v, BASE_DIR) for k, v in saved.items()},
    }
    out_json = os.path.join(args.out, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n[OK] metrik tersimpan   : {out_json}")
    for k, v in saved.items():
        print(f"[OK] gambar {k:<14}: {v}")
    print("\nLangkah berikutnya: python scripts/generate_report.py")


if __name__ == "__main__":
    main()
