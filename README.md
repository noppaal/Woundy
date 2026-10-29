# Deteksi dan Klasifikasi Jenis Luka pada Kulit Menggunakan Algoritma YOLOv8 sebagai Sistem Rekomendasi Pertolongan Pertama

**Peneliti:** Ahmad Naufal Ramadhan (NIM: 103012300239)  
**Calon Pembimbing:** Dr. Putu Harry Gunawan, S.Si., M.Si., M.Sc. (NIP: 16860043)  
**Program Studi:** S1 Informatika, Fakultas Informatika, Universitas Telkom Bandung (2026)  
**Dataset Roboflow Universe:** [wound-no-blister-2 (w-afwxp)](https://universe.roboflow.com/w-afwxp/wound-no-blister-2)

---

## 📌 Ringkasan Sistem

Aplikasi ini dibangun berdasarkan **Proposal Tugas Akhir** yang mengintegrasikan kecerdasan buatan (Computer Vision berbasis **YOLOv8**) dengan sistem pendukung keputusan medis berbasis aturan (**Rule-Based System**) menggunakan mekanisme penalaran **Forward Chaining** untuk memberikan panduan pertolongan pertama (P3K) yang akurat dan mudah dipahami oleh masyarakat umum.

Sistem mampu mengidentifikasi dan menangani 5 kategori kondisi kulit utama:
1. **Abrasions (Luka Lecet)** - Aturan `R001`
2. **Bruise (Luka Memar / Contusion)** - Aturan `R002`
3. **Burn (Luka Bakar)** - Aturan `R003` (Derajat 1 Ringan vs Derajat 2 / Bula Parah)
4. **Cut (Luka Sayat / Robek / Tusuk)** - Aturan `R004` (Dangkal Ringan vs Luka Dalam / Perdarahan Aktif)
5. **No Abnormality (Tanpa Kelainan / Kulit Sehat)** - Aturan `R005`

---

## 🏗️ Arsitektur dan Fitur Aplikasi

- **Studio Deteksi & Triage Interaktif**:
  - Dukungan masukan berkas citra (drag-and-drop / selector) dan kamera *real-time* (Webcam).
  - Galeri pengujian cepat (*one-click test samples*) dengan citra riil dari Roboflow dataset.
  - Visualisasi *bounding box* canggih dengan palet warna klinis dan label tingkat keyakinan (*confidence*).
  - Penyesuaian ambang batas keyakinan (*confidence slider* 10% - 90%).
  
- **Modul Rekomendasi Forward Chaining (Tabel 3.1)**:
  - Pelacakan fakta deteksi ke kode aturan (`R001 - R005`).
  - Indikator tingkat kegawatan: **Rendah**, **Sedang**, **Tinggi**, hingga **Gawat Darurat**.
  - Panduan tindakan terstruktur (*step-by-step actions*).
  - Peringatan medis kritis (*clinical warnings / DO NOT*), misalnya larangan penggunaan pasta gigi/mentega/es batu pada luka bakar, serta larangan mencabut benda tertancap pada luka tusuk.
  - Kuesioner triage sekunder interaktif untuk penyesuaian dinamis (misal: lepuhan bula, perdarahan aktif memancar).
  - Tombol darurat medis langsung (**119 Ambulans** / **112 Panggilan Darurat Nasional**).

- **Tab Evaluasi Model (Subbab 3.5)** di aplikasi Streamlit, seluruh angka dibaca dari `reports/metrics.json` (hasil perhitungan nyata, bukan angka manual):
  - Evaluasi **tingkat citra**: Akurasi, Presisi, Recall, Spesifisitas, F1 (macro/weighted/micro), Balanced Accuracy, Cohen's Kappa, MCC, interval kepercayaan 95%.
  - Rincian **TP / FP / FN / TN** per kelas dan **matriks konfusi 5×6** (kolom ke-6 = *Tidak Terdeteksi*).
  - Evaluasi **tingkat deteksi (box)**: Presisi, Recall, AP@0.5, AP@0.75, AP@0.5:0.95, mAP.
  - Analisis pendukung: sweep threshold, kurva Precision-Recall, kurva training, kesenjangan train-valid-test (overfitting), contoh citra salah klasifikasi, kecepatan inferensi terukur.
  - Laporan Bahasa Indonesia otomatis dengan rumus dan contoh perhitungan: `reports/LAPORAN_EVALUASI.md`.

- **Pipeline Pelatihan & Integrasi Dataset**:
  - Konfigurasi pelatihan YOLOv8 lokal (`scripts/train.py`).
  - Pengunduh otomatis dataset dari Roboflow Universe (`scripts/download_dataset.py`).
  - Pembersih kebocoran data antar split (`scripts/clean_leakage.py`).
  - Evaluasi lengkap (`scripts/evaluate_full.py`), laporan (`scripts/generate_report.py`), dan pemeriksa metrik di terminal (`scripts/check_metrics.py`).

---

## 🗂️ Catatan Dataset

- Dataset asli berjumlah **1.391 citra** (979 latih / 275 valid / 137 uji). Pada export Roboflow, split **train diaugmentasi ±3×** sehingga berisi **2.934 file**; valid dan test tidak diaugmentasi.
- Ditemukan **kebocoran data** (citra valid/test yang sama atau sangat mirip dengan citra latih, terdeteksi dengan *perceptual hash*). `scripts/clean_leakage.py` memindahkannya ke `dataset_removed/` (bukan menghapus; dapat dikembalikan dengan `--restore`). Jumlah akhir: **train 2.934 / valid 236 / test 118**.
- Metrik pada split *training* hanya menunjukkan kemampuan model menghafal. Kinerja sebenarnya diukur pada split **test**.

---

## 📂 Struktur Direktori Proyek

```
TA/
├── app/                          # Aplikasi Streamlit (ONNX + Forward Chaining)
│   ├── streamlit_app.py          # Antarmuka: Pemeriksaan luka | Evaluasi model | Tentang sistem
│   ├── config.yaml               # Threshold, kelas, teks aplikasi
│   ├── core/
│   │   ├── detector.py           # Inferensi YOLOv8 ONNX (onnxruntime)
│   │   ├── metrics.py            # Perhitungan metrik (confusion matrix, P/R/F1, Kappa, MCC, mAP)
│   │   └── evaluator.py          # Pemuat hasil evaluasi dari reports/metrics.json
│   └── rbs/                      # Mesin aturan Forward Chaining (rules.yaml, engine.py)
├── core/ , app.py, templates/, static/   # Versi Flask lama (opsional)
├── dataset/                      # train / valid / test (setelah dibersihkan dari kebocoran)
├── dataset_removed/              # Citra bocor yang dipindahkan (dapat dikembalikan)
├── models/
│   ├── wound_yolov8n.pt          # Bobot PyTorch
│   └── wound_yolov8n.onnx        # Model yang dipakai aplikasi
├── reports/                      # Keluaran evaluasi (metrics.json, gambar, LAPORAN_EVALUASI.md)
├── scripts/
│   ├── download_dataset.py       # Unduh dataset dari Roboflow
│   ├── clean_leakage.py          # Deteksi & pindahkan citra bocor antar split
│   ├── train.py                  # Pelatihan YOLOv8
│   ├── evaluate_full.py          # Evaluasi lengkap (citra + deteksi) -> reports/
│   ├── generate_report.py        # Laporan Markdown Bahasa Indonesia
│   ├── check_metrics.py          # Cetak metrik di terminal (--explain untuk penjelasan)
│   └── evaluate.py               # Evaluasi cepat bawaan Ultralytics (perlu PyTorch)
├── streamlit_app.py              # Entry point deployment
└── requirements.txt
```

---

## 🚀 Panduan Menjalankan Aplikasi

### 1. Menjalankan Aplikasi Streamlit
```bash
streamlit run streamlit_app.py
```

### 2. Mengunduh Dataset dari Roboflow Universe
```powershell
python scripts/download_dataset.py --api-key "MASUKKAN_API_KEY_ANDA"
```

### 3. Membersihkan Kebocoran Data antar Split
```bash
python scripts/clean_leakage.py            # dry-run: hanya melaporkan
python scripts/clean_leakage.py --apply    # memindahkan citra bocor ke dataset_removed/
python scripts/clean_leakage.py --restore  # mengembalikan semuanya
```

### 4. Melatih Model YOLOv8
```bash
python scripts/train.py --epochs 50 --batch 16 --imgsz 640 --device 0
```
*(Gunakan `--device cpu` jika tidak memiliki GPU NVIDIA CUDA).*

### 5. Mengevaluasi Model (Matriks Evaluasi)
```bash
python scripts/evaluate_full.py        # hitung metrik train/valid/test -> reports/metrics.json + gambar
python scripts/generate_report.py      # -> reports/LAPORAN_EVALUASI.md
python scripts/check_metrics.py        # tampilkan di terminal (tambahkan --explain untuk arti tiap metrik)
```
Pilihan: `--conf 0.35` (threshold citra, default sama dengan aplikasi), `--iou 0.45`, `--splits valid test`.
Tab **Evaluasi model** pada aplikasi membaca hasil yang sama.

---

## 📊 Basis Aturan Pakar (Tabel 3.1 Proposal Tugas Akhir)

| ID Aturan | Jenis Luka | Kondisi (IF) | Aksi (THEN) | Prioritas |
|---|---|---|---|---|
| **R001** | **Abrasion (Luka Lecet)** | Terdeteksi luka abrasi ringan, tidak ada perdarahan aktif | 1. Bersihkan area luka dengan air mengalir.<br>2. Oleskan antiseptik (*povidone iodine*).<br>3. Tutup dengan perban steril atau plester.<br>4. Ganti perban setiap 1-2 hari.<br>*Tidak perlu ke dokter.* | Rendah |
| **R002** | **Bruise (Luka Memar)** | Terdeteksi memar/contusion tanpa luka terbuka | 1. Kompres dengan es yang dibungkus kain 15-20 menit.<br>2. Istirahatkan bagian yang memar.<br>3. Elevasi (tinggikan) area yang terkena.<br>4. Jika nyeri hebat atau membesar, ke dokter.<br>*Perawatan mandiri.* | Rendah |
| **R003** | **Burn (Luka Bakar)** | Terdeteksi luka bakar superfisial atau dengan lepuhan | **A. Derajat 1 (Ringan - Kemerahan & Tanpa Lepuhan):**<br>1. Siram area dengan air dingin mengalir &plusmn; 20 menit.<br>2. JANGAN gunakan es batu, pasta gigi, minyak, mentega.<br>3. Oleskan gel lidah buaya / losion penyejuk.<br><br>**B. Derajat 2 (Parah - Ada Lepuhan atau Luas > 5 cm):**<br>1. Siram air dingin mengalir.<br>2. JANGAN pecahkan lepuhan secara sengaja.<br>3. Tutup longgar kain steril & segera ke faskes. | Rendah s.d. Tinggi |
| **R004** | **Cut (Luka Sayat / Robek / Tusuk)** | Terdeteksi luka akibat benda tajam, robekan, atau tusukan | **A. Luka Ringan (Panjang < 2 cm & Dangkal):**<br>1. Bersihkan air mengalir dan antiseptik.<br>2. Rapatkan tepi luka dengan plester butterfly, tutup perban.<br><br>**B. Luka Dalam / Tusuk (> 2 cm / Perdarahan Aktif):**<br>1. Tekan luka kuat-kuat kain bersih hentikan perdarahan.<br>2. JANGAN cabut benda asing jika tertancap.<br>3. Balut perban tekan dan segera ke IGD/faskes. | Sedang s.d. Tinggi |
| **R005** | **No Abnormality (Normal)** | Hasil deteksi tidak menunjukkan luka | 1. Tampilkan pesan: "Kulit tampak normal dan sehat. Tidak ditemukan indikasi luka terbuka, memar, atau luka bakar."<br>2. Rekomendasikan menjaga kebersihan kulit. | N/A |
