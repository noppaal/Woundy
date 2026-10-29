"""
Modul Perhitungan Metrik Evaluasi (Subbab 3.5 Proposal)

Seluruh fungsi di sini murni numpy (tanpa torch / ultralytics) sehingga:
  - dapat diuji secara terpisah,
  - dapat dipakai ulang oleh skrip evaluasi dan antarmuka Streamlit.

Dua tingkat evaluasi:
  1. Tingkat Citra (image-level)  : confusion matrix K x (K+1), kolom terakhir = "Tidak Terdeteksi".
  2. Tingkat Deteksi (box-level)  : Precision, Recall, AP@0.5, AP@0.5:0.95, mAP (gaya COCO/Ultralytics).
"""

from typing import Dict, List, Sequence, Tuple, Any
import numpy as np

NONE_LABEL = "Tidak Terdeteksi"


# ------------------------------------------------------------------------------
# A. METRIK TINGKAT CITRA
# ------------------------------------------------------------------------------
def build_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """
    Matriks konfusi (n_classes x (n_classes + 1)).
    Baris   = kelas aktual.
    Kolom 0..K-1 = kelas prediksi, kolom K = 'Tidak Terdeteksi' (y_pred == -1).
    """
    cm = np.zeros((n_classes, n_classes + 1), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), n_classes if p < 0 else int(p)] += 1
    return cm


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def per_class_metrics(cm: np.ndarray, labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """
    One-vs-Rest per kelas dari matriks konfusi K x (K+1).
      TP = cm[i,i]
      FN = total baris i - TP            (termasuk citra yang 'tidak terdeteksi')
      FP = total kolom i - TP
      TN = N - TP - FP - FN
    """
    k = len(labels)
    total = int(cm.sum())
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fp - fn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        npv = safe_div(tn, tn + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        out[name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "specificity": specificity, "npv": npv,
            "f1_score": f1,
            "accuracy_ovr": safe_div(tp + tn, total),
            "support": int(cm[i].sum()),
            "missed": int(cm[i, k]),            # aktual kelas i tetapi tidak terdeteksi
        }
    return out


def overall_metrics(cm: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    """Akurasi, rata-rata macro/weighted/micro, balanced accuracy, Cohen's Kappa, MCC."""
    k = len(labels)
    pc = per_class_metrics(cm, labels)
    total = int(cm.sum())
    correct = int(np.trace(cm[:, :k]))
    support = np.array([pc[l]["support"] for l in labels], dtype=float)
    w = support / support.sum() if support.sum() else np.zeros(k)

    def macro(key):  return float(np.mean([pc[l][key] for l in labels]))
    def weighted(key): return float(np.sum([pc[l][key] * w[i] for i, l in enumerate(labels)]))

    # Micro (single-label: micro-P = micro-R = micro-F1 = accuracy bila 'tidak terdeteksi' dihitung FN)
    tp_s = sum(pc[l]["tp"] for l in labels)
    fp_s = sum(pc[l]["fp"] for l in labels)
    fn_s = sum(pc[l]["fn"] for l in labels)
    micro_p = safe_div(tp_s, tp_s + fp_s)
    micro_r = safe_div(tp_s, tp_s + fn_s)

    # Cohen's Kappa (hanya pada kolom kelas + kolom 'none' sebagai kategori prediksi tambahan)
    row = cm.sum(axis=1).astype(float)
    col = cm.sum(axis=0).astype(float)
    po = correct / total if total else 0.0
    pe = float(sum(row[i] * col[i] for i in range(k)) / (total ** 2)) if total else 0.0
    kappa = safe_div(po - pe, 1 - pe)

    # MCC multikelas (Gorodkin) – kolom 'none' diperlakukan sebagai kelas prediksi ekstra
    c = float(correct)
    s = float(total)
    num = c * s - float(np.sum(row[:k] * col[:k]))
    den = np.sqrt(max(s ** 2 - float(np.sum(col ** 2)), 0.0)) * np.sqrt(max(s ** 2 - float(np.sum(row ** 2)), 0.0))
    mcc = safe_div(num, den)

    return {
        "total": total,
        "correct": correct,
        "undetected": int(cm[:, k].sum()),
        "accuracy": safe_div(correct, total),
        "balanced_accuracy": macro("recall"),
        "precision_macro": macro("precision"),
        "recall_macro": macro("recall"),
        "f1_macro": macro("f1_score"),
        "specificity_macro": macro("specificity"),
        "precision_weighted": weighted("precision"),
        "recall_weighted": weighted("recall"),
        "f1_weighted": weighted("f1_score"),
        "precision_micro": micro_p,
        "recall_micro": micro_r,
        "cohen_kappa": kappa,
        "mcc": mcc,
    }


def bootstrap_ci(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int,
                 n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> Dict[str, List[float]]:
    """Interval kepercayaan bootstrap persentil untuk Accuracy dan Macro-F1."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    if n == 0:
        return {"accuracy": [0.0, 0.0], "f1_macro": [0.0, 0.0]}
    labels = [str(i) for i in range(n_classes)]
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cm = build_confusion_matrix(yt[idx], yp[idx], n_classes)
        om = overall_metrics(cm, labels)
        accs.append(om["accuracy"])
        f1s.append(om["f1_macro"])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "accuracy": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
        "f1_macro": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> List[float]:
    """Interval Wilson 95% untuk proporsi (dipakai untuk Recall/Precision per kelas)."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def top_confusions(cm: np.ndarray, labels: Sequence[str], top_n: int = 5) -> List[Dict[str, Any]]:
    """Pasangan kelas yang paling sering tertukar (di luar diagonal & 'tidak terdeteksi')."""
    k = len(labels)
    pairs = []
    for i in range(k):
        for j in range(k):
            if i != j and cm[i, j] > 0:
                pairs.append({"actual": labels[i], "predicted": labels[j], "count": int(cm[i, j]),
                              "rate_of_actual": safe_div(cm[i, j], cm[i].sum())})
    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:top_n]


# ------------------------------------------------------------------------------
# B. METRIK TINGKAT DETEKSI (BOX-LEVEL)
# ------------------------------------------------------------------------------
def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU berpasangan antara a (N,4) dan b (M,4) dalam format xyxy (Persamaan 3.1)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Non-Maximum Suppression murni numpy (boxes xyxy)."""
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return keep


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP interpolasi 101 titik (konvensi COCO / Ultralytics)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x)) if hasattr(np, "trapezoid") \
        else float(np.trapz(np.interp(x, mrec, mpre), x))


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def detection_ap(preds: List[Dict[str, Any]], gts: List[Dict[str, Any]], n_classes: int,
                 conf_for_pr: float = 0.35) -> Dict[str, Any]:
    """
    Menghitung AP per kelas pada IoU 0.5:0.95 serta P/R pada ambang `conf_for_pr`.

    preds : list per citra  {"boxes": (N,4) xyxy, "scores": (N,), "classes": (N,)}
    gts   : list per citra  {"boxes": (M,4) xyxy, "classes": (M,)}
    """
    n_img = len(gts)
    result: Dict[str, Any] = {"per_class": {}}
    ap_all = np.zeros((n_classes, len(IOU_THRESHOLDS)))
    curves: Dict[int, Dict[str, List[float]]] = {}

    for c in range(n_classes):
        n_gt = int(sum((g["classes"] == c).sum() for g in gts))
        # Kumpulkan semua deteksi kelas c
        dets = []  # (score, img_idx, box)
        for i, p in enumerate(preds):
            m = p["classes"] == c
            for s, b in zip(p["scores"][m], p["boxes"][m]):
                dets.append((float(s), i, b))
        dets.sort(key=lambda d: -d[0])

        tp_matrix = np.zeros((len(dets), len(IOU_THRESHOLDS)), dtype=bool)
        used = [dict() for _ in IOU_THRESHOLDS]  # per iou: {(img, gt_idx): True}
        for di, (s, i, b) in enumerate(dets):
            g = gts[i]
            gm = np.where(g["classes"] == c)[0]
            if len(gm) == 0:
                continue
            ious = box_iou_matrix(b[None, :], g["boxes"][gm])[0]
            for ti, thr in enumerate(IOU_THRESHOLDS):
                # pilih GT dengan IoU tertinggi yang belum dipakai
                order = np.argsort(-ious)
                for oi in order:
                    if ious[oi] < thr:
                        break
                    key = (i, int(gm[oi]))
                    if key not in used[ti]:
                        used[ti][key] = True
                        tp_matrix[di, ti] = True
                        break

        scores = np.array([d[0] for d in dets])
        for ti in range(len(IOU_THRESHOLDS)):
            if n_gt == 0 or len(dets) == 0:
                ap_all[c, ti] = 0.0
                continue
            tpc = np.cumsum(tp_matrix[:, ti])
            fpc = np.cumsum(~tp_matrix[:, ti])
            rec = tpc / n_gt
            prec = tpc / np.maximum(tpc + fpc, 1e-9)
            ap_all[c, ti] = _compute_ap(rec, prec)
            if ti == 0:
                curves[c] = {"recall": rec.tolist()[::max(1, len(rec) // 200)],
                             "precision": prec.tolist()[::max(1, len(prec) // 200)]}

        # P / R pada ambang konfiden aplikasi (IoU 0.5)
        if len(dets):
            sel = scores >= conf_for_pr
            tp_sel = int(tp_matrix[sel, 0].sum())
            fp_sel = int(sel.sum() - tp_sel)
        else:
            tp_sel = fp_sel = 0
        fn_sel = n_gt - tp_sel

        result["per_class"][c] = {
            "n_gt": n_gt,
            "n_pred": len(dets),
            "tp": tp_sel, "fp": fp_sel, "fn": fn_sel,
            "precision": safe_div(tp_sel, tp_sel + fp_sel),
            "recall": safe_div(tp_sel, tp_sel + fn_sel),
            "ap50": float(ap_all[c, 0]),
            "ap50_95": float(ap_all[c].mean()),
            "ap75": float(ap_all[c, 5]),
        }

    pcs = result["per_class"]
    result["map50"] = float(np.mean([pcs[c]["ap50"] for c in range(n_classes)]))
    result["map75"] = float(np.mean([pcs[c]["ap75"] for c in range(n_classes)]))
    result["map50_95"] = float(np.mean([pcs[c]["ap50_95"] for c in range(n_classes)]))
    result["precision"] = float(np.mean([pcs[c]["precision"] for c in range(n_classes)]))
    result["recall"] = float(np.mean([pcs[c]["recall"] for c in range(n_classes)]))
    p, r = result["precision"], result["recall"]
    result["f1"] = safe_div(2 * p * r, p + r)
    result["pr_curves"] = curves
    return result
