"""
Generator Laporan Evaluasi (Bahasa Indonesia)

Membaca reports/metrics.json (hasil scripts/evaluate_full.py) dan menulis
reports/LAPORAN_EVALUASI.md berisi: definisi, rumus, contoh perhitungan dengan
angka asli, tabel per split & per kelas, matriks konfusi, interpretasi klinis,
analisis overfitting, analisis threshold, kecepatan, dan keterbatasan.

Pemakaian:  python scripts/generate_report.py
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
NONE_LABEL = "Tidak Terdeteksi"

SPLIT_ORDER = [("training", "Training"), ("validasi", "Validasi"), ("testing", "Testing")]
CLASS_ADVICE = {
    "Abrasions": "luka lecet ringan - risiko klinis rendah; kesalahan lebih bersifat kosmetik/perawatan",
    "Bruise": "memar - risiko rendah, namun bisa menutupi cedera dalam",
    "Burn": "luka bakar - KRITIS: salah mengira derajat berat sebagai ringan menunda rujukan medis",
    "Cut": "luka sayat/tusuk - KRITIS: luka dalam yang terlewat berisiko perdarahan & infeksi",
    "No Abnormality": "kulit normal - kesalahan berarti menolak penanganan pada kulit yang sebenarnya terluka (FN pada kelas lain) atau alarm palsu",
}


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def ci_str(ci):
    return f"[{ci[0] * 100:.1f}% - {ci[1] * 100:.1f}%]"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def cm_table(ev):
    cm = ev["confusion_matrix"]
    cols = cm["columns"]
    rows = []
    for i, lab in enumerate(cm["labels"]):
        row = cm["matrix"][i]
        cells = [f"**{v}**" if j == i else str(v) for j, v in enumerate(row)]
        rows.append([f"**{lab}**"] + cells + [sum(row)])
    tot = [sum(cm["matrix"][i][j] for i in range(len(cm["labels"]))) for j in range(len(cols))]
    rows.append(["**Total prediksi**"] + [f"**{t}**" for t in tot] + [f"**{sum(tot)}**"])
    return md_table(["Aktual \\ Prediksi"] + cols + ["Total"], rows)


def main():
    path = os.path.join(REPORT_DIR, "metrics.json")
    if not os.path.exists(path):
        sys.exit("[ERROR] reports/metrics.json belum ada. Jalankan: python scripts/evaluate_full.py")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)

    meta, ds, tm, evs = d["meta"], d["dataset"], d["timing"], d["evaluations"]
    figs = {k: v.replace("\\", "/").replace("reports/", "") for k, v in d.get("figures", {}).items()}
    L = []
    A = L.append

    test = evs.get("testing") or list(evs.values())[-1]
    t_ov, t_pc = test["metrics_overall"], test["metrics_per_class"]
    t_det = test["detection"]

    A("# Laporan Evaluasi Model Deteksi & Klasifikasi Luka Kulit (YOLOv8)")
    A("")
    A(f"> Dibuat otomatis oleh `scripts/generate_report.py` pada **{meta['generated_at']}**. "
      "Seluruh angka dihitung langsung dari model — **bukan** nilai yang diketik manual.")
    A("")
    A(md_table(["Item", "Nilai"], [
        ["Model", f"`{meta['model']}` (model yang sama dengan aplikasi)"],
        ["Threshold keyakinan evaluasi citra", f"{meta['conf_threshold']} (sama dengan status *Terdeteksi* di aplikasi)"],
        ["IoU NMS", meta["iou_nms"]],
        ["Aturan label citra (ground truth)", meta["image_truth_rule"]],
        ["Aturan prediksi citra", meta["image_pred_rule"]],
        ["Perangkat", f"{meta['device']} — {meta['platform']}"],
    ]))
    A("")

    # ---------------- 1. Dataset
    A("## 1. Dataset yang Dievaluasi")
    A("")
    rows = []
    for sp, nm in SPLIT_ORDER:
        if sp in evs:
            key = {"training": "train", "validasi": "valid", "testing": "test"}[sp]
            cc = ds["image_class_counts"].get(key, {})
            rows.append([nm, ds["splits"].get(key, "-")] + [cc.get(c, 0) for c in ds["class_names"]])
    A(md_table(["Split", "Jumlah citra"] + ds["class_names"], rows))
    A("")
    rem = ds.get("removed_for_leakage") or {}
    A("**Catatan metodologis penting:**")
    A("")
    A(f"- {ds['note_train']} Metrik pada split *training* sebaiknya hanya dipakai untuk melihat *seberapa baik model "
      "menghafal*, bukan sebagai ukuran kinerja.")
    if rem:
        A(f"- **Pembersihan kebocoran data (*data leakage*)**: {rem.get('valid', 0)} citra validasi dan {rem.get('test', 0)} citra uji "
          "yang identik/sangat mirip dengan citra latih (perceptual hash, jarak Hamming ≤ 4) dikeluarkan "
          "(disimpan di `dataset_removed/`). Tanpa langkah ini, akurasi uji akan terlihat lebih tinggi dari kenyataan.")
    A("- Label citra diturunkan dari anotasi *bounding box* (kelas dominan), karena dataset berbasis deteksi objek.")
    A("")

    # ---------------- 2. Definisi
    A("## 2. Definisi TP, FP, FN, TN pada Konteks Luka")
    A("")
    A("Evaluasi dilakukan **satu kelas terhadap sisanya** (*One-vs-Rest*). Contoh untuk kelas **Cut (luka sayat)**:")
    A("")
    A(md_table(["Istilah", "Arti", "Contoh (kelas Cut)"], [
        ["**TP** (True Positive)", "Citra kelas X diprediksi X", "Luka sayat → diprediksi *Cut* ✔"],
        ["**FP** (False Positive)", "Citra bukan X diprediksi X", "Luka lecet → diprediksi *Cut* (alarm salah)"],
        ["**FN** (False Negative)", "Citra kelas X tidak diprediksi X (salah kelas **atau** tidak terdeteksi)",
         "Luka sayat → diprediksi *Bruise*, atau tidak ada deteksi (bahaya: pengguna tidak diberi panduan)"],
        ["**TN** (True Negative)", "Citra bukan X dan tidak diprediksi X", "Memar → diprediksi *Bruise* (bukan *Cut*) ✔"],
    ]))
    A("")
    A(f"> Kolom **{NONE_LABEL}** pada matriks konfusi menampung citra yang skor deteksinya di bawah threshold. "
      "Itu dihitung sebagai **FN** untuk kelas aslinya sehingga *Recall* tidak dibesar-besarkan.")
    A("")

    # ---------------- 3. Rumus
    A("## 3. Rumus Metrik & Cara Membacanya")
    A("")
    A(md_table(["Metrik", "Rumus", "Pertanyaan yang dijawab", "Prioritas pada P3K"], [
        ["Accuracy", "(TP+TN) / N", "Dari semua citra, berapa yang benar?", "Gambaran umum (menyesatkan bila kelas tak seimbang)"],
        ["Precision", "TP / (TP+FP)", "Saat model bilang kelas X, seberapa sering benar?", "Mengurangi alarm/panduan yang salah"],
        ["Recall (Sensitivitas)", "TP / (TP+FN)", "Dari semua kasus X nyata, berapa yang ditemukan?", "**Tertinggi** — kasus terlewat berbahaya"],
        ["Specificity", "TN / (TN+FP)", "Dari semua non-X, berapa yang benar dikenali sebagai non-X?", "Menghindari penanganan berlebihan"],
        ["NPV", "TN / (TN+FN)", "Saat model bilang bukan X, seberapa sering benar?", "Kepercayaan pada hasil negatif"],
        ["F1-Score", "2·P·R / (P+R)", "Keseimbangan Precision & Recall", "Ringkasan per kelas"],
        ["Macro avg", "rata-rata sederhana per kelas", "Semua kelas dianggap sama penting", "Utama (adil untuk kelas kecil)"],
        ["Weighted avg", "rata-rata berbobot jumlah sampel", "Mencerminkan komposisi data", "Pembanding"],
        ["Balanced Accuracy", "rata-rata Recall per kelas", "Akurasi yang tidak bias kelas besar", "Pembanding"],
        ["Cohen's Kappa", "(po − pe) / (1 − pe)", "Kesepakatan melebihi kebetulan (0–1)", "Validasi kekokohan"],
        ["MCC", "korelasi prediksi–aktual multikelas (−1…1)", "Metrik tunggal paling ketat", "Validasi kekokohan"],
        ["IoU", "Area(A∩B) / Area(A∪B)", "Seberapa pas kotak prediksi dengan kotak asli", "Kualitas lokalisasi"],
        ["AP / mAP", "luas di bawah kurva Precision–Recall, dirata-rata antar kelas", "Kualitas deteksi di semua threshold keyakinan", "Metrik standar YOLO"],
    ]))
    A("")
    A("**mAP@0.5** = AP dengan kotak dianggap benar bila IoU ≥ 0.5. **mAP@0.5:0.95** = rata-rata AP pada IoU 0.50, 0.55, …, 0.95 "
      "(jauh lebih ketat). Interval kepercayaan 95%: *bootstrap* 2.000× untuk Accuracy & Macro-F1, *Wilson* untuk Recall/Precision per kelas.")
    A("")

    # ---------------- 4. Contoh perhitungan
    A("## 4. Contoh Perhitungan Langkah demi Langkah (Data Uji)")
    A("")
    labels = test["confusion_matrix"]["labels"]
    worst = min(labels, key=lambda c: t_pc[c]["f1_score"])
    w = t_pc[worst]
    N = t_ov["total"]
    A(f"Kelas dengan F1 terendah pada data uji: **{worst}** (N total = {N}).")
    A("")
    A("```text")
    A(f"TP = {w['tp']}   FP = {w['fp']}   FN = {w['fn']}   TN = {N} - {w['tp']} - {w['fp']} - {w['fn']} = {w['tn']}")
    P = w["precision"]; R = w["recall"]
    A(f"Precision   = TP/(TP+FP) = {w['tp']}/({w['tp']}+{w['fp']}) = {P:.4f}")
    A(f"Recall      = TP/(TP+FN) = {w['tp']}/({w['tp']}+{w['fn']}) = {R:.4f}")
    A(f"Specificity = TN/(TN+FP) = {w['tn']}/({w['tn']}+{w['fp']}) = {w['specificity']:.4f}")
    A(f"F1-Score    = 2·P·R/(P+R) = 2·{P:.4f}·{R:.4f}/({P:.4f}+{R:.4f}) = {w['f1_score']:.4f}")
    A(f"Accuracy keseluruhan = diagonal/N = {t_ov['correct']}/{N} = {t_ov['accuracy']:.4f}")
    A("```")
    A("")

    # ---------------- 5. Ringkasan per split
    A("## 5. Ringkasan Metrik per Split")
    A("")
    rows = []
    for sp, nm in SPLIT_ORDER:
        if sp not in evs:
            continue
        e = evs[sp]; o = e["metrics_overall"]; dd = e["detection"]["overall"]
        rows.append([nm, o["total"], pct(o["accuracy"]), pct(o["balanced_accuracy"]), pct(o["precision_macro"]),
                     pct(o["recall_macro"]), pct(o["f1_macro"]), f"{o['cohen_kappa']:.3f}", f"{o['mcc']:.3f}",
                     o["undetected"], f"{dd['map50']:.3f}", f"{dd['map50_95']:.3f}"])
    A(md_table(["Split", "N", "Accuracy", "Bal. Acc", "Precision (macro)", "Recall (macro)", "F1 (macro)",
                "Kappa", "MCC", "Tdk terdeteksi", "mAP@0.5", "mAP@0.5:0.95"], rows))
    A("")
    A(f"**Data uji — interval kepercayaan 95%:** Accuracy {pct(t_ov['accuracy'], 1)} {ci_str(t_ov['ci95']['accuracy'])}, "
      f"Macro-F1 {pct(t_ov['f1_macro'], 1)} {ci_str(t_ov['ci95']['f1_macro'])}.")
    A("")
    A(f"> Karena data uji hanya **{N} citra**, rentang CI cukup lebar. Satu citra salah mengubah akurasi ≈ {100 / N:.2f} poin. "
      "Klaim kinerja sebaiknya memakai rentang ini, bukan satu angka.")
    A("")
    if "cm_combined" in figs:
        A(f"![Matriks konfusi gabungan]({figs['cm_combined']})")
        A("")

    # ---------------- 6. Per kelas + CM tiap split
    for sp, nm in SPLIT_ORDER:
        if sp not in evs:
            continue
        e = evs[sp]
        A(f"## 6.{[s for s, _ in SPLIT_ORDER].index(sp) + 1} Rincian Split {nm}")
        A("")
        A("**Matriks konfusi (jumlah citra)** — diagonal tebal = benar:")
        A("")
        A(cm_table(e))
        A("")
        rows = []
        for c in e["confusion_matrix"]["labels"]:
            m = e["metrics_per_class"][c]
            rows.append([f"{m['label_id']} ({c})", m["support"], m["tp"], m["fp"], m["fn"], m["tn"],
                         pct(m["precision"], 1), f"{pct(m['recall'], 1)} {ci_str(m['recall_ci95'])}",
                         pct(m["specificity"], 1), pct(m["f1_score"], 1), m["missed"]])
        A(md_table(["Kelas", "Support", "TP", "FP", "FN", "TN", "Precision", "Recall [CI95]", "Specificity", "F1", "Tdk terdeteksi"], rows))
        A("")
        if e["top_confusions"]:
            tc = "; ".join(f"**{x['actual']} → {x['predicted']}**: {x['count']} citra ({pct(x['rate_of_actual'], 1)} dari kelas aktual)"
                           for x in e["top_confusions"][:3])
            A(f"Pasangan tertukar terbanyak: {tc}.")
            A("")
        rows = []
        for c, m in e["detection"]["per_class"].items():
            rows.append([c, m["n_gt"], f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['ap50']:.3f}", f"{m['ap75']:.3f}", f"{m['ap50_95']:.3f}"])
        o = e["detection"]["overall"]
        rows.append(["**Semua kelas**", "", f"**{o['precision']:.3f}**", f"**{o['recall']:.3f}**", f"**{o['map50']:.3f}**",
                     f"**{o['map75']:.3f}**", f"**{o['map50_95']:.3f}**"])
        A("**Evaluasi tingkat deteksi (box-level)** — P/R pada threshold aplikasi, IoU 0.5:")
        A("")
        A(md_table(["Kelas", "Jumlah box GT", "Precision", "Recall", "AP@0.5", "AP@0.75", "AP@0.5:0.95"], rows))
        A("")
        key = {"training": "cm_training", "validasi": "cm_valid", "testing": "cm_test"}[sp]
        A(f"![Matriks konfusi {nm}](confusion_matrix_{sp}.png)")
        A("")

    # ---------------- 7. mAP vs accuracy
    A("## 7. Mengapa Ada Dua Jenis Metrik?")
    A("")
    A("| | Evaluasi **tingkat citra** | Evaluasi **tingkat deteksi (box)** |")
    A("|---|---|---|")
    A("| Pertanyaan | Apakah kelas luka pada foto ini benar? | Apakah lokasi **dan** kelas luka tepat? |")
    A("| Relevansi | Langsung menentukan aturan P3K (R001–R005) yang ditampilkan | Mengukur kualitas model deteksi YOLO itu sendiri |")
    A("| Metrik | Accuracy, P/R/F1, Kappa, MCC | AP, mAP@0.5, mAP@0.5:0.95 |")
    A("| Sensitif pada | Threshold keyakinan | Kualitas kotak (IoU) |")
    A("")
    A("Keduanya dilaporkan agar hasil dapat dibandingkan dengan literatur YOLO (mAP) **dan** relevan dengan fungsi sistem rekomendasi (klasifikasi citra).")
    A("")

    # ---------------- 8. Interpretasi
    A("## 8. Interpretasi Hasil Data Uji")
    A("")
    best = max(labels, key=lambda c: t_pc[c]["f1_score"])
    low_rec = min(labels, key=lambda c: t_pc[c]["recall"])
    low_prec = min(labels, key=lambda c: t_pc[c]["precision"])
    A(f"- **Kelas terbaik:** {best} (F1 {pct(t_pc[best]['f1_score'], 1)}). **Terlemah:** {worst} (F1 {pct(t_pc[worst]['f1_score'], 1)}).")
    A(f"- **Recall terendah:** {low_rec} ({pct(t_pc[low_rec]['recall'], 1)}) → {CLASS_ADVICE.get(low_rec, '')}.")
    A(f"- **Precision terendah:** {low_prec} ({pct(t_pc[low_prec]['precision'], 1)}).")
    undet = t_ov["undetected"]
    A(f"- **Citra tidak terdeteksi:** {undet} dari {N} ({pct(undet / max(N, 1), 1)}). "
      "Pada aplikasi, kasus ini ditampilkan sebagai *Kemungkinan*/*Tidak terdeteksi* sehingga pengguna tidak diberi panduan keliru dengan percaya diri.")
    if test["top_confusions"]:
        tc0 = test["top_confusions"][0]
        A(f"- **Kesalahan dominan:** *{tc0['actual']}* sering dikira *{tc0['predicted']}* ({tc0['count']} citra). "
          "Secara visual kelas-kelas ini memang mirip (warna kemerahan, tepi tidak tegas), sehingga disarankan menambah data pada pasangan kelas ini.")
    A("")
    A("**Implikasi klinis:** dalam sistem P3K, *Recall* kelas **Burn** dan **Cut** paling kritis karena kegagalan mendeteksinya dapat menunda "
      "rujukan. Karena itu sistem memakai *kuesioner triage sekunder* dan peringatan medis pada aturan R003/R004, sehingga keputusan akhir "
      "tidak bergantung semata pada keluaran model.")
    A("")
    if "misclassified" in figs:
        A(f"![Contoh citra uji salah klasifikasi]({figs['misclassified']})")
        A("")
        A("Contoh kesalahan ditampilkan untuk membantu analisis kualitatif (pencahayaan, sudut, luka berukuran kecil, atau label ambigu).")
        A("")
    if "per_class" in figs:
        A(f"![Metrik per kelas]({figs['per_class']})")
        A("")

    # ---------------- 9. Overfitting
    A("## 9. Analisis Overfitting (Kesenjangan Antar Split)")
    A("")
    if "training" in evs and "testing" in evs:
        tr = evs["training"]["metrics_overall"]; te = t_ov
        gap = tr["accuracy"] - te["accuracy"]
        trm = evs["training"]["detection"]["overall"]["map50_95"]; tem = t_det["overall"]["map50_95"]
        A(md_table(["Metrik", "Training", "Testing", "Selisih"], [
            ["Accuracy citra", pct(tr["accuracy"]), pct(te["accuracy"]), f"{gap * 100:+.2f} poin"],
            ["Macro-F1 citra", pct(tr["f1_macro"]), pct(te["f1_macro"]), f"{(tr['f1_macro'] - te['f1_macro']) * 100:+.2f} poin"],
            ["mAP@0.5:0.95", f"{trm:.3f}", f"{tem:.3f}", f"{trm - tem:+.3f}"],
        ]))
        A("")
        verdict = ("**tinggi** → indikasi overfitting; pertimbangkan augmentasi tambahan, regularisasi, atau data lebih banyak."
                   if gap > 0.10 else "sedang." if gap > 0.05 else "kecil → generalisasi baik.")
        A(f"Kesenjangan akurasi training–testing sebesar **{gap * 100:.1f} poin** tergolong {verdict}")
        A("")
        A("![Kesenjangan generalisasi](" + figs.get("gap", "generalization_gap.png") + ")")
        A("")
    if "training" in figs:
        A("Riwayat training (dari log `results.csv`):")
        A("")
        A(f"![Kurva training]({figs['training']})")
        A("")

    # ---------------- 10. Threshold
    A("## 10. Analisis Pemilihan Threshold Keyakinan")
    A("")
    sweep = d.get("threshold_sweep", {}).get("test") or []
    if sweep:
        sel = [r for r in sweep if round(r["threshold"] * 100) % 10 == 5 or abs(r["threshold"] - 0.35) < 1e-6]
        sel = sorted({r["threshold"]: r for r in sel}.values(), key=lambda r: r["threshold"])
        A(md_table(["Threshold", "Accuracy", "Macro-F1", "Precision (macro)", "Recall (macro)", "% Tdk terdeteksi"],
                   [[f"{r['threshold']:.2f}" + (" ⬅ aplikasi" if abs(r["threshold"] - 0.35) < 1e-6 else ""),
                     pct(r["accuracy"], 1), pct(r["f1_macro"], 1), pct(r["precision_macro"], 1),
                     pct(r["recall_macro"], 1), pct(r["undetected_rate"], 1)] for r in sel]))
        A("")
        bf = max(sweep, key=lambda r: r["f1_macro"])
        A(f"Macro-F1 tertinggi pada data uji terjadi di threshold **{bf['threshold']:.2f}** ({pct(bf['f1_macro'], 1)}). "
          "Semakin tinggi threshold, citra yang tidak terdeteksi bertambah (Recall turun, Precision naik). "
          "Threshold aplikasi (0.35) dipilih sebagai kompromi: lebih rendah akan menaikkan alarm salah, lebih tinggi akan melewatkan kasus. "
          "**Catatan:** pemilihan threshold sebaiknya dipertimbangkan pada data **validasi**, bukan data uji, agar tidak bias.")
        A("")
        A("![Sweep threshold](" + figs.get("sweep", "threshold_sweep.png") + ")")
        A("")
    if "pr" in figs:
        A(f"![Kurva PR]({figs['pr']})")
        A("")

    # ---------------- 11. Kecepatan
    A("## 11. Kecepatan Inferensi (Terukur)")
    A("")
    A(md_table(["Tahap", "Rata-rata (ms)"], [
        ["Pra-pemrosesan (letterbox)", f"{tm['pre_ms']:.1f}"],
        ["Inferensi ONNX", f"{tm['infer_ms']:.1f} (median {tm['infer_ms_median']:.1f}; P95 {tm['infer_ms_p95']:.1f})"],
        ["Pasca-pemrosesan (NMS)", f"{tm['post_ms']:.1f}"],
        ["**Total per citra**", f"**{tm['total_ms']:.1f}** (≈ {tm['fps']:.1f} FPS)"],
    ]))
    A("")
    A(f"Diukur pada: {meta['device']}, {meta['platform']}, CPU saja (onnxruntime). Nilai bergantung perangkat.")
    A("")

    # ---------------- 12. Keterbatasan
    A("## 12. Keterbatasan")
    A("")
    A(f"1. Data uji kecil ({N} citra) → interval kepercayaan lebar; perbedaan beberapa persen belum tentu signifikan.")
    A("2. Label citra diturunkan dari kelas box dominan; citra multi-lesi campuran dapat ambigu.")
    A("3. Kemiripan antar split sudah dibersihkan secara otomatis (perceptual hash) namun tidak menjamin 100% bebas dari kemiripan semantik "
      "(mis. pasien/foto sumber yang sama dengan pencahayaan berbeda).")
    A("4. Dataset gabungan dari beberapa sumber → bias domain; kinerja pada foto kamera pengguna nyata (pencahayaan, kulit gelap, ukuran luka) belum teruji.")
    A("5. Model dilatih sebelum pembersihan kebocoran; sebagian citra validasi yang dibersihkan sempat dipakai saat pemilihan bobot terbaik. "
      "Melatih ulang pada split bersih akan memberi estimasi yang lebih ketat.")
    A("6. Sistem bukan alat diagnosis; hasil adalah panduan P3K awal.")
    A("")
    A("---")
    A("*Reproduksi:* `python scripts/evaluate_full.py` → `python scripts/generate_report.py`")

    out = os.path.join(REPORT_DIR, "LAPORAN_EVALUASI.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[OK] Laporan tersimpan: {out}")


if __name__ == "__main__":
    main()
