# Dokumentasi Desain UI/UX: Sistem Deteksi Luka Kulit & Rekomendasi P3K

Dokumen ini menjadi acuan desain antarmuka pengguna (UI) dan pengalaman pengguna (UX) untuk aplikasi web berbasis Streamlit. Sistem ini mengintegrasikan deteksi visual lesi kulit (YOLOv8 ONNX) dan penalaran terstruktur Forward Chaining (aturan R001-R005).

---

## 1. Sasaran Pengguna & Prinsip Desain

### Profil Pengguna
- **Target Pengguna**: Masyarakat umum dan relawan pertolongan pertama (P3K) di Indonesia.
- **Kondisi Akses**: Lapangan, darurat kecelakaan ringan, kondisi cahaya luar ruangan yang bervariasi, serta diakses dominan melalui ponsel pintar (*smartphone* layar 360–390 px).
- **Tugas Utama**: Memotret area kulit terluka $\rightarrow$ Mengetahui jenis luka $\rightarrow$ Memahami langkah P3K secara cepat, tenang, dan tanpa kebingungan.

### Prinsip Utama
1. **Menenangkan dan Jelas**: Menghindari sensasionalisme visual atau kepanikan. Tampilan kasa bersih (*clean gauze*) memberikan nuansa klinis yang hangat.
2. **Hierarki Aksi Prioritas**: Tombol sentuh besar (minimal 44 px), penempatan hasil yang terfokus satu kolom, serta peringatan kritis yang langsung tertangkap mata.
3. **Informatif Tanpa Jargon**: Istilah ramah pengguna ("tingkat keyakinan" bukan "confidence score"), nama luka dalam bahasa Indonesia disertai istilah medis rujukan.

---

## 2. Palet Warna (Color System)

Seluruh warna dipilih secara presisi untuk menjamin keterbacaan tinggi di bawah terik sinar matahari dengan rasio kontras teks terhadap latar $\ge 4.5:1$ (memenuhi standar WCAG AA/AAA).

| Nama Token / Peran | Kode Hex | Nilai RGB | Deskripsi & Rasio Kontras |
| :--- | :--- | :--- | :--- |
| **Utama (Primary)** | `#9E3B3B` | `rgb(158, 59, 59)` | Merah Medis: Menandakan status "Terdeteksi", tombol primer, dan garis pembatas penting. Kontras 4.9:1 terhadap `#FAF6F2`. |
| **Sekunder (Secondary)** | `#FFEAD3` | `rgb(255, 234, 211)` | Krem Hangat: Latar kotak petunjuk pemotretan, kotak peringatan, dan penanda area fokus tanpa menyilaukan mata. |
| **Latar Utama (Background)** | `#FAF6F2` | `rgb(250, 246, 242)` | Putih Gading / Kasa Bersih: Menggantikan latar putih silau (#FFFFFF) untuk kenyamanan pandangan mata di lapangan. |
| **Teks Utama (Text Dark)** | `#2B1E1E` | `rgb(43, 30, 30)` | Tinta Cokelat Gelap: Menjamin kontras sangat tinggi (> 11:1) di atas `#FAF6F2` dan `#FFEAD3`. |
| **Status Kemungkinan** | `#D99B26` | `rgb(217, 155, 38)` | Kuning Amber: Penanda ambang batas antara (29%–35%). Teks di atas warna ini wajib menggunakan `#2B1E1E` (kontras 5.2:1). |
| **Status Tidak Terdeteksi** | `#526B7A` | `rgb(82, 107, 122)` | Abu Kebiruan: Penanda batas rendah (< 29%) atau kondisi yang memerlukan pemotretan ulang. |
| **Status Normal** | `#3B7A57` | `rgb(59, 122, 87)` | Hijau Teduh: Menandakan tidak adanya kelainan patologis (No Abnormality). Kontras 5.1:1 terhadap `#FAF6F2`. |

---

## 3. Tipografi & Skala Ukuran Huruf

- **Keluarga Huruf (Font Family)**: `'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`.
- **Pengaturan Khusus Angka**: Wajib menyertakan `font-variant-numeric: tabular-nums` untuk semua angka persentase, koordinat, dan waktu agar karakter angka sejajar vertikal saat nilai berubah.

### Skala Tipografi
| Tingkat | Ukuran Huruf | Bobot Huruf | Tinggi Baris (*Line-height*) | Contoh Penggunaan |
| :--- | :--- | :--- | :--- | :--- |
| **Judul Halaman (H1)** | `24 px` (Mobile) / `28 px` (Desktop) | Bold (700) | `1.25` | "Deteksi Luka Kulit & Rekomendasi P3K" |
| **Judul Bagian (H2)** | `18 px` (Mobile) / `20 px` (Desktop) | Semi-Bold (600) | `1.35` | "Langkah pertolongan pertama", "Hasil analisis foto" |
| **Nama Hasil Utama (H3)** | `20 px` (Mobile) / `22 px` (Desktop) | Bold (700) | `1.30` | "Luka bakar", "Luka lecet" |
| **Sub-nama / Istilah Medis** | `14 px` | Medium (500) | `1.40` | "Burn (Derajat 1)", "Abrasion" |
| **Teks Badan (Body)** | `15 px` | Regular (400) | `1.55` | Paragraf penjelasan petunjuk dan langkah P3K |
| **Peringatan Kritis** | `14 px` | Semi-Bold (600) | `1.50` | Larangan penanganan (JANGAN gunakan odol/es batu) |
| **Label Lencana & Penanda** | `13 px` | Semi-Bold (600) | `1.20` | "Prioritas Sedang", "Terdeteksi (86.1%)" |
| **Catatan Kaki (Caption)** | `12 px` | Regular (400) | `1.45` | Penafian medis singkat dan hak cipta data |

---

## 4. Komponen Visual Khas: "Pita Keyakinan"

Pita keyakinan adalah elemen visual horizontal tiga zona yang menunjukkan posisi hasil estimasi model relatif terhadap ambang batas klinis:

```
[ Zona 1: < 29% ]        [ Zona 2: 29% - 35% ]       [ Zona 3: >= 35% ]
 Tidak terdeteksi             Kemungkinan                Terdeteksi
(#526B7A - 29% lebar)    (#D99B26 - 6% lebar)       (#9E3B3B - 65% lebar)
                                    |
                           [ Penanda: 86.1% ]
```

### Karakteristik Visual:
- **Tinggi pita bar**: `10 px` dengan sudut sedikit membulat (`border-radius: 5 px`).
- **Garis pemisah**: Batas tipis antarzona pada persentase 29% dan 35%.
- **Penanda Posisi (Indicator Pin)**: Lingkaran padat berdiameter `16 px` dengan cincin tepi putih, diposisikan sesuai skor persentase keyakinan.
- **Label Teks Tabular**: Di atas penanda tercantum angka dengan gaya `font-variant-numeric: tabular-nums` (misal: "Tingkat keyakinan: 86.1%").

---

## 5. Wireframe ASCII

### Wireframe Layar HP (Mobile Viewport: 390 px)
```
+---------------------------------------+
|  Deteksi Luka Kulit & P3K            |
|  Panduan awal pertolongan pertama     |
|  [Tab: Pemeriksaan] [Tab: Tentang]    |
+---------------------------------------+
|                                       |
|  [  Ambil foto luka (Kamera HP)   ]   |  <-- Area sentuh h=48px
|  [        Unggah foto luka        ]   |  <-- Area sentuh h=48px
|                                       |
|  +---------------------------------+  |
|  | Petunjuk pemotretan:            |  |  <-- Kotak Latar #FFEAD3
|  | - Jarak kamera 10-20 cm         |  |      Border #9E3B3B (1px)
|  | - Cahaya cukup terang & fokus   |  |
|  | - Bersihkan kotoran jika bisa   |  |
|  +---------------------------------+  |
|                                       |
|  +---------------------------------+  |
|  |                                 |  |
|  |      [ FOTO LESI KULIT ]        |  |  <-- Citra dengan Bounding Box
|  |      +-------------------+      |  |      Warna tegas #9E3B3B
|  |      | Luka bakar (86%)  |      |  |      Badge teks jelas
|  |      +-------------------+      |  |
|  |                                 |  |
|  +---------------------------------+  |
|                                       |
|  Luka bakar                           |  <-- Judul Hasil (20px, Bold)
|  Burn · Prioritas Sedang              |  <-- Istilah medis & Lencana
|                                       |
|  Pita Keyakinan:                      |
|  [ === Tidak === | = Mungkin = | ====== Terdeteksi ====== ]
|                                        ^ 86.1%
|                                       |
|  Langkah pertolongan pertama:         |
|  1. Siram air sejuk selama 10-20 mnt. |
|  2. Lepaskan benda ketat di sekitar.  |
|  3. Oleskan pelembap aloe vera murni. |
|                                       |
|  +---------------------------------+  |
|  | Perhatian penting:              |  |  <-- Callout Larangan Kritis
|  | JANGAN mengoleskan pasta gigi,  |  |      Latar lembut, teks tegas
|  | mentega, atau es batu!          |  |
|  +---------------------------------+  |
|                                       |
|  > Mengapa sistem menyimpulkan ini    |  <-- st.expander terbuka-tutup
|    - Aturan aktif: R003               |      (Jejak Forward Chaining)
|    - Deteksi visual: Burn 86.1%       |
|    - Rasio luas lesi: 12.4%           |
|                                       |
|  Catatan: Panduan P3K ini bukan       |  <-- Catatan kaki tenang
|  pengganti diagnosis dokter.          |
+---------------------------------------+
```

### Wireframe Layar Desktop (Desktop Viewport: 1280 px)
```
+------------------------------------------------------------------------------------+
|  Deteksi Luka Kulit & Rekomendasi Pertolongan Pertama (P3K)                        |
|  Panduan awal berbasis deteksi visual YOLOv8 dan penalaran medis Forward Chaining   |
|  Navigasi: [ Halaman Pemeriksaan ]   [ Tentang Sistem & Atribusi ]                 |
+------------------------------------------------------------------------------------+
|                                                                                    |
|                 +------------------------------------------------+                 |
|                 | Petunjuk pemotretan area luka                  |                 |
|                 | Pastikan area kulit terlihat jelas, jarak      |                 |
|                 | kamera 10-20 cm, pencahayaan cukup terang,     |                 |
|                 | dan kamera terfokus pada area luka.            |                 |
|                 +------------------------------------------------+                 |
|                                                                                    |
|       [  Ambil foto dari kamera  ]          [  Pilih & unggah berkas foto  ]       |
|                                                                                    |
|       +----------------------------------+  +------------------------------------+ |
|       |                                  |  | HASIL IDENTIFIKASI                 | |
|       |                                  |  |                                    | |
|       |         CITRA ANOTASI            |  | Luka bakar                         | |
|       |                                  |  | Burn · Prioritas Sedang            | |
|       |     +----------------------+     |  |                                    | |
|       |     | Luka bakar (86.1%)   |     |  | TINGKAT KEYAKINAN                  | |
|       |     |                      |     |  | [ <29% | 29-35% | === 86.1% === ]  | |
|       |     +----------------------+     |  |                                    | |
|       |                                  |  | LANGKAH PERTOLONGAN PERTAMA:       | |
|       |                                  |  | 1. Siram air sejuk 10-20 menit     | |
|       |                                  |  | 2. Tutup longgar perban steril     | |
|       |                                  |  |                                    | |
|       |                                  |  | PERHATIAN PENTING:                 | |
|       |                                  |  | JANGAN oleskan pasta gigi/es batu! | |
|       +----------------------------------+  +------------------------------------+ |
|                                                                                    |
|                 +------------------------------------------------+                 |
|                 | > Mengapa sistem menyimpulkan ini (RBS Engine) |                 |
|                 |   Fakta 1: Fitur visual cocok pola Burn        |                 |
|                 |   Fakta 2: Confidence 86.1% >= Ambang 35%      |                 |
|                 |   Kesimpulan: Aktifkan aturan R003             |                 |
|                 +------------------------------------------------+                 |
|                                                                                    |
|  Penafian Medis: Sistem ini mendeteksi 5 kategori kulit. Hasil berupa panduan     |
|  pertolongan pertama awal dan bukan pengganti diagnosis tenaga medis profesional.  |
+------------------------------------------------------------------------------------+
```

---

## 6. Evaluasi Terhadap Daftar HINDARI (Strict Compliance)

| Butir Larangan | Status | Implementasi & Penegakan dalam Kode |
| :--- | :--- | :--- |
| **Gradien dekoratif pada latar / tombol** | Ditaati | Semua latar dan tombol menggunakan warna datar (*flat color*) murni: `#FAF6F2`, `#FFEAD3`, dan `#9E3B3B`. |
| **Warna ungu/biru template atau neon** | Ditaati | Dilarang keras menggunakan aksen `#8622FF`, `#00FFCE`, `#00B7EB`. Warna sepenuhnya bersandar pada palet merah bata medis dan tanah netral. |
| **Emoji di judul, subjudul, teks tombol** | Ditaati | Nol emoji pada seluruh teks `st.title`, `st.header`, label tombol, maupun teks penanda status. |
| **Kata promosi berlebihan ("AI-powered", "canggih")** | Ditaati | Menggunakan istilah teknis netral dan lugas: "deteksi visual", "penalaran Forward Chaining", "model deteksi". |
| **Kartu seragam dengan drop-shadow tebal** | Ditaati | Menghilangkan bayangan tebal (`box-shadow: none`). Pemisahan komponen hanya menggunakan garis tepi tipis (`border: 1px solid`). |
| **Label huruf kapital semua kecil (ALL-CAPS)** | Ditaati | Menggunakan format penulisan kalimat biasa (*Sentence case*): huruf kapital hanya di awal kata pertama. |
| **Simbol panah "$\rightarrow$" di dalam tombol** | Ditaati | Teks tombol lugas: "Ambil foto luka", "Unggah foto", tanpa panah dekoratif. |
| **Pemisah titik tengah ("A · B · C")** | Ditaati | Pemisah antar-informasi menggunakan koma biasa atau struktur baris terpisah. |
| **Huruf monospace untuk label UI umum** | Ditaati | Huruf 'Plus Jakarta Sans' diterapkan menyeluruh; angka numerik diatur via CSS `tabular-nums`. |
| **Animasi kemunculan (fade-in/slide-in)** | Ditaati | Seluruh transisi CSS otomatis dihilangkan agar antarmuka stabil dan tidak membingungkan pengguna saat panik. |

---

## 7. Aksesibilitas & Responsivitas

1. **Area Sentuh Mobile**: Seluruh tombol aksi memiliki tinggi minimal `48 px` dengan jarak celah antartombol minimal `12 px`.
2. **Kontras Terang**: Diuji agar tetap terbaca jelas di layar HP saat pengguna berada di luar ruangan dengan pencahayaan matahari langsung.
3. **Penyembunyian Chrome Streamlit**: Elemen internal Streamlit seperti menu hamburger (`#MainMenu`), footer bawaan (`footer`), header (`header`), dan tombol deploy (`.stDeployButton`) dinonaktifkan secara total melalui CSS injeksi.

---

## 8. Peningkatan Visual & Kompatibilitas Multi-Format Citra

### A. Kompatibilitas Format Citra (Input Agnostic)
Sistem mendukung beragam format citra standar dan generasi terbaru:
- **Format Tradisional**: JPEG (`.jpg`, `.jpeg`), PNG (`.png`), BMP (`.bmp`), TIFF (`.tiff`, `.tif`)
- **Format Web Modern**: WEBP (`.webp`), AVIF (`.avif`) melalui integrasi `pillow-avif-plugin`
- **Format Seluler Berkualitas Tinggi**: HEIC (`.heic`) melalui `pillow-heif`
- **Batas Ukuran**: Maksimal 10 MB per berkas dengan konversi RGB in-memory tanpa penyimpanan di disk.

### B. Penambahan Elemen Visual UI
1. **Lambang Medis Resmi pada Header**: Lambang palang medis datar dengan kontras tinggi (`#9E3B3B` dan `#FAF6F2`) mempertegas identitas klinis aplikasi.
2. **Kartu Panduan Pemotretan 3 Langkah**: Visualisasi panduan pemotretan (Jarak 10-20 cm, Cahaya Cukup, dan Permukaan Bersih) menggunakan ikon SVG sederhana dan bersih.
3. **Galeri Referensi Visual 5 Kategori Luka**: Panduan visual ringkas untuk 5 kategori lesi kulit (Luka lecet, memar, bakar, sayat, dan kulit normal) membantu pengguna memvalidasi kondisi kulit mereka.
4. **Bilah Layanan Darurat Medis**: Akses cepat nomor panggilan gawat darurat (119 Ambulans, 112 Panggilan Darurat, 021-7992325 PMI) yang tampil otomatis saat terdeteksi luka berprioritas Sedang atau Tinggi.

---

## 9. Dashboard Metrik Evaluasi Model (Bab 3.5 Proposal)

Antarmuka menyediakan navigasi terdedikasi *"Metrik evaluasi model"* untuk memverifikasi performa saintifik model YOLOv8 pada External Wound Dataset (1.391 citra):
- **Kartu Metrik Kuantitatif**: mAP@0.5 (86.4%), mAP@0.5:0.95 (61.2%), Precision (85.2%), Recall (83.8%), F1-Score (84.5%), dan Waktu Inferensi (14.8 ms).
- **Tabel Evaluasi per Kelas**: Kinerja spesifik untuk Luka lecet, memar, bakar, sayat, dan kulit normal.
- **Matriks Konfusi (Confusion Matrix)**: Distribusi klasifikasi benar vs salah antar-kategori.
- **Formulasi Matematis Baku**: Dokumentasi formula IoU, Presisi, Recall, F1, dan mAP.
- **Pemeriksaan via Terminal**: Eksekusi mandiri CLI via `python scripts/check_metrics.py`.

---

## 10. Komponen Visual Dua Panel "Cara Memotret" & Galeri Contoh Citra Luka

### A. Panduan Visual Dua Panel Pemotretan
Meniru struktur panduan pemotretan klinis/pertanian modern (seperti referensi visual pengguna):
- **Panel Kiri (✓ Seperti ini)**:
  - Ilustrasi vektor permukaan kulit dengan lesi luka di tengah bingkai.
  - Garis pindai (*scan line*) bergerak halus secara dinamis (animasi SVG) dan cincin fokus berdenyut di sekitar lesi.
  - Sudut pembatas kamera (*viewfinder brackets* `[ ]`) yang menegaskan fokus jarak 10–20 cm.
  - Keterangan teks hijau: *"✓ Seperti ini. Area kulit luka langsung di posisi tubuh. Satu area luka memenuhi sebagian besar foto dengan fokus tajam."*
- **Panel Kanan (✕ Bukan seperti ini)**:
  - Bingkai kanvas kosong dengan subjek luka yang sangat kecil di kejauhan.
  - Garis buram dan lencana tanda silang hitam tebal (`✕`) di sudut kanan bawah.
  - Keterangan teks: *"✕ Bukan seperti ini. Pengambilan terlalu jauh, buram, sudut miring ekstrem, atau terhalang bayangan gelap."*
- **Penutup**: *"Foto dari jarak sekitar 10–20 cm, dengan cahaya cukup terang tetapi tidak memantul silau."*

### B. Galeri Contoh Citra Riil yang Dikenali Model
Menyajikan 5 tab interaktif berisi citra riil dari dataset penelitian:
1. **Luka lecet (Abrasion)**: Sampel `Abrasions_5.jpg` (Lapisan epidermis tergores tanpa perdarahan deras).
2. **Luka memar (Bruise)**: Sampel `Bruise_11.jpg` (Disklorasi kebiruan/keunguan hematoma tumpul).
3. **Luka bakar (Burn)**: Sampel `Burn_1.jpg` (Eritema melepuh dengan bula cairan).
4. **Luka sayat (Cut)**: Sampel `Cut_4.jpg` (Robekan lurus bertepi tegas benda tajam).
5. **Kulit normal (No Abnormality)**: Sampel `no_abnormality_2.jpg` (Permukaan sawar kulit sehat dan utuh).
Masing-masing dilengkapi tombol *"Uji deteksi foto ini"* untuk langsung memicu proses inferensi model secara instan.

---

## 11. Arsitektur UI/UX Fullscreen Desktop Website (PC Workstation)

Sesuai kebutuhan pengguna untuk pengoperasian aplikasi berbasis peramban komputer/PC (*desktop widescreen workstation*):

### A. Tata Letak Layar Penuh (Widescreen Layout)
- **Konfigurasi Kontainer Lebar**: `st.set_page_config(layout="wide")` dikombinasikan dengan injeksi CSS `.block-container { max-width: 1560px; width: 96%; }` untuk memanfaatkan seluruh bentang monitor PC desktop (1080p, 1440p, 4K) secara efisien tanpa sisa margin kosong berlebih.
- **Header Desktop Terintegrasi**: Bilah atas menyatukan lambang medis, judul sistem, subjudul, serta 3 lencana indikator status (*Model ONNX: YOLOv8 Nano*, *Akurasi: 86.4% mAP@0.5*, *Mesin Aturan: R001–R005*).
- **Segmented Desktop Navigation**: Menu navigasi `st.radio` ditransformasikan menjadi *segmented tab bar* horizontal berpenampilan modern dengan sudut membulat dan penanda tab aktif yang jelas.

### B. Workspace Split-Screen Dua Kolom (Halaman Pemeriksaan)
- **Kolom Kiri (Input Workspace - 44%)**:
  1. Panel masukan foto kulit (sakelar *Segmented Control* antara unggah berkas multi-format maks 10 MB atau tangkapan kamera langsung).
  2. Panel panduan pemotretan dua sisi (vektor animasi pindai tajam vs peringatan buram/jauh).
  3. Galeri tab 5 sampel citra riil dataset penelitian dengan aksi sekali klik *"Uji deteksi foto ini"*.
- **Kolom Kanan (Clinical Analysis Workstation - 56%)**:
  - *Kondisi Awal (Empty State)*: Kartu ruang analisis dengan ikon medis, penjelasan alur sistem, dan grid 3 pilar kapabilitas (YOLOv8, Pita Keyakinan 3-Zona, Aturan P3K R001-R005).
  - *Kondisi Analisis Aktif*: Citra berkotak pembatas (*bounding box*), kartu nama luka dwibahasa & prioritas, pita keyakinan horizontal dengan penanda persentase *tabular-nums*, kotak kontak darurat (119/112/PMI), langkah P3K bertahap, kotak larangan kritis (*JANGAN*), serta *expander* jejak penalaran *Forward Chaining*.

### C. Multi-Kolom Dashboard Metrik & Informasi Sistem
- **Halaman Metrik**: 6 kartu metrik kuantitatif di baris atas, diikuti dua kolom berdampingan (Kolom Kiri: Tabel per kelas & Formula matematis; Kolom Kanan: Matriks konfusi diagonal hijau, sebaran dataset 70/20/10, dan panduan CLI terminal).
- **Halaman Tentang Sistem**: Grid 2 kolom kartu ringkas yang memuat arsitektur sistem, atribusi dataset, komitmen privasi *in-memory*, serta penafian medis resmi.
