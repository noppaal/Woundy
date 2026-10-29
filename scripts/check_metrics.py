"""
Skrip Pemeriksaan Metrik Kuantitatif Model YOLOv8 (Bab 3.5 Proposal Tugas Akhir)
Hanya menampilkan metrik utama: Accuracy, Precision, Recall, F1-Score, dan mAP.
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")
W = 85


def pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def print_split_metrics(title: str, total_images: int, ev: dict):
    ov = ev["metrics_overall"]
    det = ev["detection"]["overall"]
    pcm = ev["metrics_per_class"]
    pcd = ev["detection"]["per_class"]

    print("=" * W)
    print(f"  {title.upper()} (TOTAL: {total_images} CITRA)")
    print("=" * W)

    # 1. Ringkasan Metrik Keseluruhan
    acc = pct(ov.get("accuracy", 0.0))
    p = pct(ov.get("precision_macro", 0.0))
    r = pct(ov.get("recall_macro", 0.0))
    f1 = pct(ov.get("f1_macro", 0.0))
    map50 = pct(det.get("map50", 0.0))
    map50_95 = pct(det.get("map50_95", 0.0))

    print("\n1. RINGKASAN METRIK KESELURUHAN (OVERALL PERFORMANCE):")
    print("-" * W)
    print(f"  Akurasi (Overall Accuracy)            : {acc:>7}")
    print(f"  Presisi (Macro Precision)             : {p:>7}")
    print(f"  Sensitivitas (Macro Recall)           : {r:>7}")
    print(f"  F1-Score (Macro F1)                   : {f1:>7}")
    print(f"  Mean Average Precision (mAP@0.5)      : {map50:>7}")
    print(f"  Mean Average Precision (mAP@0.5:0.95) : {map50_95:>7}")
    print("-" * W)

    # 2. Performa per Kategori Luka
    print("\n2. PERFORMA PER KATEGORI LUKA:")
    print("-" * W)
    header = f"  {'Kategori Luka':<31} | {'Akurasi':<8} | {'Presisi':<8} | {'Recall':<8} | {'F1-Score':<8} | {'mAP@0.5':<8}"
    print(header)
    print("-" * W)

    for c_name, m in pcm.items():
        lbl = f"{m.get('label_id', c_name)} ({c_name})"
        c_acc = pct(m.get("accuracy_ovr", 0.0))
        c_p = pct(m.get("precision", 0.0))
        c_r = pct(m.get("recall", 0.0))
        c_f1 = pct(m.get("f1_score", 0.0))
        c_ap = pct(pcd.get(c_name, {}).get("ap50", 0.0))

        row = f"  {lbl:<31} | {c_acc:>8} | {c_p:>8} | {c_r:>8} | {c_f1:>8} | {c_ap:>8}"
        print(row)

    print("-" * W)
    print()


def main():
    parser = argparse.ArgumentParser(description="Pemeriksaan Metrik Model YOLOv8")
    parser.add_argument("--split", choices=["training", "validasi", "testing"], default=None,
                        help="Tampilkan hanya split tertentu (default: semua split)")
    args = parser.parse_args()

    if not os.path.exists(METRICS_PATH):
        sys.exit(f"[ERROR] Berkas metrik {METRICS_PATH} tidak ditemukan. Jalankan: python scripts/evaluate_full.py")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    evals = data.get("evaluations", {})

    splits_config = [
        ("training", "1. Matrix Evaluasi (Training Set)"),
        ("validasi", "2. Matrix Evaluasi (Validation Set)"),
        ("testing", "3. Matrix Evaluasi (Testing Set)")
    ]

    for key, title in splits_config:
        if key in evals and (args.split is None or args.split == key):
            ev = evals[key]
            total_img = ev.get("total_images", 0)
            print_split_metrics(title, total_img, ev)


if __name__ == "__main__":
    main()
