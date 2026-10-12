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

- **Dashboard Evaluasi Model & Metrik Riset (Bab 3.5)**:
  - Skor performa kuantitatif: mAP@0.5, mAP@0.5:0.95, Precision, Recall, dan F1-Score.
  - Grafik interaktif performa per kelas dan distribusi dataset (Chart.js).
  - Dokumentasi matematis formula evaluasi (IoU, mAP, F1-Score).

- **Pipeline Pelatihan & Integrasi Dataset**:
  - Konfigurasi pelatihan YOLOv8 lokal (`scripts/train.py`).
  - Pengunduh otomatis dataset 1.391 citra dari Roboflow Universe (`scripts/download_dataset.py`).
  - Skrip pengujian dan evaluasi mandiri (`scripts/evaluate.py`).

---

## 📂 Struktur Direktori Proyek

```
c:\OPENCODE/
├── core/
│   ├── detector.py             # Pipeline deteksi YOLOv8 & computer vision visualizer
│   ├── rule_engine.py          # Forward Chaining Rule-Based System (Tabel 3.1)
│   └── evaluator.py            # Mesin kalkulasi metrik (mAP, IoU, Precision, Recall, F1)
├── dataset/
│   └── data.yaml               # Konfigurasi kelas dataset YOLOv8
├── models/
│   └── wound_yolov8n.pt        # Bobot model YOLOv8
├── sample_images/              # Citra sampel riil dari dataset Roboflow
├── scripts/
│   ├── download_dataset.py     # Unduh dataset 1.391 citra dari Roboflow API
│   ├── train.py                # Pelatihan YOLOv8 (imgsz 640, epochs 50, batch 16)
│   └── evaluate.py             # Evaluasi model pada test split (137 citra)
├── static/
│   ├── css/
│   │   └── style.css           # Desain UI modern medical healthtech
│   └── js/
│       └── app.js              # Kontroler antarmuka, kamera, dan API async
├── templates/
│   └── index.html              # Antarmuka web utama
├── app.py                      # Server backend Flask & REST API
├── requirements.txt            # Dependensi Python
└── README.md                   # Dokumentasi lengkap
```

---

## 🚀 Panduan Menjalankan Aplikasi

### 1. Menjalankan Server Web Lokal
Jalankan perintah berikut di terminal:
```bash
python app.py
```
Aplikasi akan aktif di peramban pada alamat:
👉 **`http://127.0.0.1:5000`**

### 2. Mengunduh Dataset Penuh dari Roboflow Universe (1.391 Citra)
Cukup jalankan perintah tanpa parameter, lalu masukkan API Key Anda saat diminta:
```powershell
python scripts/download_dataset.py
```
Atau sertakan API Key secara langsung (gunakan tanda petik dua):
```powershell
python scripts/download_dataset.py --api-key "MASUKKAN_API_KEY_ANDA"
```
*(Atau gunakan tombol unduh langsung pada menu tab **Dataset & Pelatihan** di aplikasi web `http://127.0.0.1:5000`)*.

### 3. Melatih Model YOLOv8
Setelah dataset terunduh ke folder `dataset/`:
```bash
python scripts/train.py --epochs 50 --batch 16 --imgsz 640 --device 0
```
*(Gunakan `--device cpu` jika tidak memiliki GPU NVIDIA CUDA)*.

### 4. Mengevaluasi Model pada Test Set
```bash
python scripts/evaluate.py --model models/wound_yolov8n.pt --split test
```

---

## 📊 Basis Aturan Pakar (Tabel 3.1 Proposal Tugas Akhir)

| ID Aturan | Jenis Luka | Kondisi (IF) | Aksi (THEN) | Prioritas |
|---|---|---|---|---|
| **R001** | **Abrasion (Luka Lecet)** | Terdeteksi luka abrasi ringan, tidak ada perdarahan aktif | 1. Bersihkan area luka dengan air mengalir.<br>2. Oleskan antiseptik (*povidone iodine*).<br>3. Tutup dengan perban steril atau plester.<br>4. Ganti perban setiap 1-2 hari.<br>*Tidak perlu ke dokter.* | Rendah |
| **R002** | **Bruise (Luka Memar)** | Terdeteksi memar/contusion tanpa luka terbuka | 1. Kompres dengan es yang dibungkus kain 15-20 menit.<br>2. Istirahatkan bagian yang memar.<br>3. Elevasi (tinggikan) area yang terkena.<br>4. Jika nyeri hebat atau membesar, ke dokter.<br>*Perawatan mandiri.* | Rendah |
| **R003** | **Burn (Luka Bakar)** | Terdeteksi luka bakar superfisial atau dengan lepuhan | **A. Derajat 1 (Ringan - Kemerahan & Tanpa Lepuhan):**<br>1. Siram area dengan air dingin mengalir &plusmn; 20 menit.<br>2. JANGAN gunakan es batu, pasta gigi, minyak, mentega.<br>3. Oleskan gel lidah buaya / losion penyejuk.<br><br>**B. Derajat 2 (Parah - Ada Lepuhan atau Luas > 5 cm):**<br>1. Siram air dingin mengalir.<br>2. JANGAN pecahkan lepuhan secara sengaja.<br>3. Tutup longgar kain steril & segera ke faskes. | Rendah s.d. Tinggi |
| **R004** | **Cut (Luka Sayat / Robek / Tusuk)** | Terdeteksi luka akibat benda tajam, robekan, atau tusukan | **A. Luka Ringan (Panjang < 2 cm & Dangkal):**<br>1. Bersihkan air mengalir dan antiseptik.<br>2. Rapatkan tepi luka dengan plester butterfly, tutup perban.<br><br>**B. Luka Dalam / Tusuk (> 2 cm / Perdarahan Aktif):**<br>1. Tekan luka kuat-kuat kain bersih hentikan perdarahan.<br>2. JANGAN cabut benda asing jika tertancap.<br>3. Balut perban tekan dan segera ke IGD/faskes. | Sedang s.d. Tinggi |
| **R005** | **No Abnormality (Normal)** | Hasil deteksi tidak menunjukkan luka | 1. Tampilkan pesan: "Kulit tampak normal dan sehat. Tidak ditemukan indikasi luka terbuka, memar, atau luka bakar."<br>2. Rekomendasikan menjaga kebersihan kulit. | N/A |
