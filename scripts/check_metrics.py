"""
Skrip Pemeriksaan Metrik Kuantitatif Model YOLOv8 (Bab 3.5 Proposal Tugas Akhir)
External Wound Dataset (1.391 citra Roboflow Universe & Kaggle)
"""

import sys
import os

# Tambahkan folder app ke sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(BASE_DIR, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(1, BASE_DIR)

from core.evaluator import BENCHMARK_RESULTS

def display_metrics():
    ds = BENCHMARK_RESULTS["dataset_info"]
    ov = BENCHMARK_RESULTS["metrics_overall"]
    cls_m = BENCHMARK_RESULTS["metrics_per_class"]
    cm = BENCHMARK_RESULTS["confusion_matrix"]

    print("=" * 72)
    print("  LAPORAN HASIL METRIK EVALUASI MODEL YOLOV8 (BAB 3.5 PROPOSAL)")
    print(f"  Dataset: {ds['name']}")
    print(f"  Total Citra: {ds['total_images']} | Train: {ds['splits']['train']} (70%) | Valid: {ds['splits']['valid']} (20%) | Test: {ds['splits']['test']} (10%)")
    print("=" * 72)

    print("\n1. RINGKASAN METRIK KESELURUHAN (OVERALL PERFORMANCE):")
    print("-" * 72)
    print(f"  Mean Average Precision (mAP@0.5)      : {ov['mAP_50'] * 100:.2f}%  (IoU Threshold = 0.50)")
    print(f"  Mean Average Precision (mAP@0.5:0.95) : {ov['mAP_50_95'] * 100:.2f}%  (Rata-rata 10 IoU multi-ambang)")
    print(f"  Presisi (Precision)                   : {ov['precision'] * 100:.2f}%  (Ketepatan deteksi positif)")
    print(f"  Sensitivitas (Recall)                 : {ov['recall'] * 100:.2f}%  (Cakupan lesi kulit terdeteksi)")
    print(f"  F1-Score                              : {ov['f1_score'] * 100:.2f}%  (Harmonic mean Precision & Recall)")
    print(f"  Kecepatan Inferensi                   : {ov['inference_speed_ms']} ms/frame  (~{ov['fps']} FPS)")
    print("-" * 72)

    print("\n2. PERFORMA DETEKSI PER KATEGORI LUKA:")
    print("-" * 72)
    header = f"{'Kategori Luka':<22} | {'Presisi':<8} | {'Recall':<8} | {'F1-Score':<8} | {'mAP@0.5':<8} | {'Sampel Uji':<10}"
    print(header)
    print("-" * 72)
    for c_name, m in cls_m.items():
        lbl = f"{m.get('label_id', c_name)} ({c_name})"
        row = f"{lbl:<22} | {m['precision']*100:>7.1f}% | {m['recall']*100:>7.1f}% | {m['f1_score']*100:>7.1f}% | {m['mAP_50']*100:>7.1f}% | {m['instances']:>10}"
        print(row)
    print("-" * 72)

    print("\n3. MATRIKS KONFUSI (CONFUSION MATRIX DATA UJI):")
    print("-" * 72)
    labels = cm["labels"]
    lbl_abbr = ["Abras", "Bruise", "Burn", "Cut", "Normal", "Bkgnd"]
    print(f"{'Aktual \\ Pred':<14} | " + " | ".join([f"{l:>6}" for l in lbl_abbr]))
    print("-" * 72)
    for i, row in enumerate(cm["matrix"]):
        row_str = " | ".join([f"{v:>6}" for v in row])
        actual_name = lbl_abbr[i] if i < len(lbl_abbr) else f"C{i}"
        print(f"{actual_name:<14} | {row_str}")
    print("-" * 72)
    print("=" * 72)

if __name__ == "__main__":
    display_metrics()
