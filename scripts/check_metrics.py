"""
Skrip Pemeriksaan Metrik Kuantitatif Model YOLOv8 (Bab 3.5 Proposal Tugas Akhir)

Membaca reports/metrics.json (hasil `python scripts/evaluate_full.py`) dan mencetak:
  1. Matrix Evaluasi (Training)
  2. Matrix Evaluasi (Validasi)
  3. Matrix Evaluasi (Testing)
masing-masing berisi ringkasan, metrik per kelas (TP/FP/FN/TN), matriks konfusi,
metrik tingkat deteksi (mAP), serta penjelasan arti tiap angka.

Pemakaian:
    python scripts/check_metrics.py            # semua split
    python scripts/check_metrics.py --split testing
    python scripts/check_metrics.py --explain  # tampilkan penjelasan rumus
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(BASE_DIR, "reports", "metrics.json")
W = 100


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def line(ch="-"):
    print("  " + ch * (W - 4))


def explain():
    print("=" * W)
    print("  PENJELASAN METRIK")
    print("=" * W)
    rows = [
        ("TP", "Citra kelas X diprediksi X"),
        ("FP", "Citra BUKAN X tetapi diprediksi X (alarm salah)"),
        ("FN", "Citra kelas X tidak diprediksi X (salah kelas ATAU tidak terdeteksi)"),
        ("TN", "Citra bukan X dan tidak diprediksi X"),
        ("Presisi", "TP/(TP+FP)  -> saat model menyebut X, seberapa sering benar"),
        ("Recall", "TP/(TP+FN)  -> dari semua X nyata, berapa yang ditemukan (KRITIS untuk Burn & Cut)"),
        ("Spesifisitas", "TN/(TN+FP)  -> dari semua non-X, berapa yang benar dikenali sebagai non-X"),
        ("F1", "2*P*R/(P+R) -> keseimbangan Presisi dan Recall"),
        ("Macro", "rata-rata sederhana antar kelas (adil untuk kelas kecil)"),
        ("Weighted", "rata-rata berbobot jumlah sampel kelas"),
        ("Kappa", "kesepakatan melebihi kebetulan (0-1; >0.8 sangat baik)"),
        ("MCC", "korelasi prediksi-aktual multikelas (-1..1), metrik tunggal paling ketat"),
        ("CI95", "interval kepercayaan 95% (bootstrap/Wilson); lebar bila data sedikit"),
        ("IoU", "Area(A irisan B)/Area(A gabungan B) -> kecocokan kotak prediksi dan kotak asli"),
        ("AP / mAP", "luas di bawah kurva Presisi-Recall (dirata-rata antar kelas)"),
        ("mAP@0.5:0.95", "mAP dirata-rata pada IoU 0.50..0.95; jauh lebih ketat daripada mAP@0.5"),
    ]
    for k, v in rows:
        print(f"  {k:<14}: {v}")
    print()


def show_split(idx, key, ev):
    ov, pcm, det, cm = ev["metrics_overall"], ev["metrics_per_class"], ev["detection"], ev["confusion_matrix"]
    labels = cm["labels"]
    print("=" * W)
    print(f"  {idx}. {ev['name'].upper()}  -  {ev['split_name']}  (N = {ov['total']} citra)")
    print("=" * W)

    ci = ov["ci95"]
    print("  A. RINGKASAN TINGKAT CITRA (apakah kelas luka pada foto benar?)")
    line()
    print(f"     Akurasi              : {pct(ov['accuracy'], 2):>8}  ({ov['correct']} dari {ov['total']} benar)   CI95 [{pct(ci['accuracy'][0])} - {pct(ci['accuracy'][1])}]")
    print(f"     Balanced Accuracy    : {pct(ov['balanced_accuracy'], 2):>8}  (rata-rata Recall per kelas)")
    print(f"     Presisi (macro)      : {pct(ov['precision_macro'], 2):>8}   | weighted {pct(ov['precision_weighted'])} | micro {pct(ov['precision_micro'])}")
    print(f"     Recall  (macro)      : {pct(ov['recall_macro'], 2):>8}   | weighted {pct(ov['recall_weighted'])} | micro {pct(ov['recall_micro'])}")
    print(f"     F1-Score (macro)     : {pct(ov['f1_macro'], 2):>8}   | weighted {pct(ov['f1_weighted'])}   CI95 [{pct(ci['f1_macro'][0])} - {pct(ci['f1_macro'][1])}]")
    print(f"     Spesifisitas (macro) : {pct(ov['specificity_macro'], 2):>8}")
    print(f"     Cohen's Kappa        : {ov['cohen_kappa']:>8.4f}")
    print(f"     MCC                  : {ov['mcc']:>8.4f}")
    print(f"     Tidak terdeteksi     : {ov['undetected']:>8}   (skor < threshold; dihitung FN)")
    line()

    print("\n  B. METRIK PER KELAS (one-vs-rest)")
    line()
    hdr = f"     {'Kelas':<20}|{'Smpl':>5}|{'TP':>5}|{'FP':>5}|{'FN':>5}|{'TN':>6}|{'Presisi':>9}|{'Recall':>8}|{'Spes.':>7}|{'F1':>7}|{'TdkDet':>7}"
    print(hdr)
    line()
    for c in labels:
        m = pcm[c]
        print(f"     {m['label_id'] + ' (' + c + ')':<20.20}|{m['support']:>5}|{m['tp']:>5}|{m['fp']:>5}|{m['fn']:>5}|{m['tn']:>6}|"
              f"{pct(m['precision']):>9}|{pct(m['recall']):>8}|{pct(m['specificity']):>7}|{pct(m['f1_score']):>7}|{m['missed']:>7}")
    line()
    print("     Recall CI95 (Wilson): " + "; ".join(f"{c}: [{pct(pcm[c]['recall_ci95'][0], 0)}-{pct(pcm[c]['recall_ci95'][1], 0)}]" for c in labels))

    print("\n  C. MATRIKS KONFUSI (baris = aktual, kolom = prediksi)")
    line()
    cols = cm["labels_abbr"] + ["Tdk Det"]
    print(f"     {'Aktual \\ Pred':<14}|" + "|".join(f"{c:>8}" for c in cols) + f"|{'Total':>7}")
    line()
    mat = cm["matrix"]
    csum = [0] * len(cols)
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            csum[j] += v
        cells = "|".join((f"[{v:>5}] " if j == i else f"{v:>8}") for j, v in enumerate(row))
        print(f"     {cm['labels_abbr'][i]:<14}|{cells}|{sum(row):>7}")
    line()
    print(f"     {'Total prediksi':<14}|" + "|".join(f"{v:>8}" for v in csum) + f"|{sum(csum):>7}")
    for x in ev.get("top_confusions", [])[:3]:
        print(f"     Tertukar: {x['actual']} -> {x['predicted']} = {x['count']} citra ({pct(x['rate_of_actual'])} dari kelas aktual)")

    o = det["overall"]
    print("\n  D. TINGKAT DETEKSI / BOX (IoU 0.5; P & R pada threshold aplikasi)")
    line()
    print(f"     {'Kelas':<16}|{'Box GT':>8}|{'Presisi':>9}|{'Recall':>8}|{'AP@0.5':>8}|{'AP@0.75':>9}|{'AP@.5:.95':>10}")
    line()
    for c, m in det["per_class"].items():
        print(f"     {c:<16}|{m['n_gt']:>8}|{m['precision']:>9.3f}|{m['recall']:>8.3f}|{m['ap50']:>8.3f}|{m['ap75']:>9.3f}|{m['ap50_95']:>10.3f}")
    line()
    print(f"     {'SEMUA (mAP)':<16}|{'':>8}|{o['precision']:>9.3f}|{o['recall']:>8.3f}|{o['map50']:>8.3f}|{o['map75']:>9.3f}|{o['map50_95']:>10.3f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["training", "validasi", "testing"], default=None)
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(METRICS):
        sys.exit("[ERROR] reports/metrics.json belum ada. Jalankan dahulu: python scripts/evaluate_full.py")
    with open(METRICS, encoding="utf-8") as fh:
        d = json.load(fh)

    meta, ds, tm = d["meta"], d["dataset"], d["timing"]
    print("=" * W)
    print("  LAPORAN METRIK EVALUASI MODEL YOLOv8 (BAB 3.5 PROPOSAL TUGAS AKHIR)")
    print(f"  Model        : {meta['model']}  |  dibuat {meta['generated_at']}")
    print(f"  Threshold    : citra {meta['conf_threshold']}  |  IoU NMS {meta['iou_nms']}")
    print(f"  Jumlah citra : " + ", ".join(f"{k}={v}" for k, v in ds["splits"].items()))
    rem = ds.get("removed_for_leakage")
    if rem:
        print(f"  Dibersihkan  : {rem.get('valid', 0)} citra valid & {rem.get('test', 0)} citra test (mirip data latih) -> dataset_removed/")
    print(f"  Kecepatan    : {tm['total_ms']:.1f} ms/citra (~{tm['fps']:.1f} FPS, CPU)")
    print("=" * W + "\n")

    if args.explain:
        explain()

    order = [("training", 1), ("validasi", 2), ("testing", 3)]
    for key, idx in order:
        if key in d["evaluations"] and (args.split in (None, key)):
            show_split(idx, key, d["evaluations"][key])

    print("  Laporan lengkap  : reports/LAPORAN_EVALUASI.md  (python scripts/generate_report.py)")
    print("  Penjelasan metrik: python scripts/check_metrics.py --explain")


if __name__ == "__main__":
    main()
