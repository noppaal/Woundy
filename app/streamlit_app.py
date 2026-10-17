"""
Antarmuka Web Streamlit: Deteksi Luka Kulit & Rekomendasi Pertolongan Pertama (P3K)
YOLOv8 ONNX + Forward Chaining Rule-Based System (R001 - R005)
Mendukung format: JPG, JPEG, PNG, WEBP, AVIF, BMP, TIFF, HEIC.
"""

import os
import sys
from io import BytesIO
import yaml
import numpy as np
from PIL import Image
import streamlit as st
import textwrap

# Registrasi plugin format citra lanjutan (AVIF & HEIC)
try:
    import pillow_avif
except ImportError:
    pass

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Pastikan path modul app/ dapat dimuat
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(1, PARENT_DIR)

# Impor mesin inferensi, aturan, dan metrik evaluasi
try:
    from core.detector import WoundONNXDetector
    from rbs.engine import ForwardChainingEngine
    from core.evaluator import BENCHMARK_RESULTS
except ImportError:
    from app.core.detector import WoundONNXDetector
    from app.rbs.engine import ForwardChainingEngine
    from app.core.evaluator import BENCHMARK_RESULTS

# -----------------------------------------------------------------------------
# Konfigurasi Halaman Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Luka Kulit & Rekomendasi P3K",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Muat Konfigurasi dari config.yaml
# -----------------------------------------------------------------------------
@st.cache_resource
def load_app_config():
    config_candidates = [
        os.path.join(CURRENT_DIR, "config.yaml"),
        os.path.join(PARENT_DIR, "app", "config.yaml"),
        "config.yaml",
        "app/config.yaml"
    ]
    for cp in config_candidates:
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {}

APP_CONFIG = load_app_config()

def safe_markdown(html_content: str):
    """Renders HTML safely into Streamlit without triggering CommonMark indented code block parsing."""
    cleaned = "\n".join(line.strip() for line in html_content.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)

def resolve_sample_path(rel_path: str) -> str:
    """Resolves relative file path across possible working directories."""
    candidates = [
        rel_path,
        os.path.join(CURRENT_DIR, rel_path),
        os.path.join(PARENT_DIR, rel_path),
        os.path.join(CURRENT_DIR, "..", rel_path)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return rel_path


# -----------------------------------------------------------------------------
# Inisialisasi Model & Mesin Penalaran (Hemat Memori dengan cache_resource)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_detector():
    model_candidates = [
        os.path.join(PARENT_DIR, "models", "wound_yolov8n.onnx"),
        os.path.join(CURRENT_DIR, "models", "wound_yolov8n.onnx"),
        os.path.join(CURRENT_DIR, "..", "models", "wound_yolov8n.onnx"),
        "models/wound_yolov8n.onnx"
    ]
    chosen_path = None
    for p in model_candidates:
        if os.path.exists(p):
            chosen_path = p
            break
    config_path = os.path.join(CURRENT_DIR, "config.yaml")
    return WoundONNXDetector(model_path=chosen_path, config_path=config_path)

@st.cache_resource(show_spinner=False)
def get_rule_engine():
    rules_candidates = [
        os.path.join(CURRENT_DIR, "rbs", "rules.yaml"),
        os.path.join(PARENT_DIR, "app", "rbs", "rules.yaml"),
        "app/rbs/rules.yaml"
    ]
    chosen_rules = None
    for p in rules_candidates:
        if os.path.exists(p):
            chosen_rules = p
            break
    return ForwardChainingEngine(rules_path=chosen_rules)

# -----------------------------------------------------------------------------
# Injeksi Gaya CSS Kustom (Strict Compliance terhadap Panduan Desain)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* Reset dan Aturan Tipografi Global */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #FAF6F2 !important;
    color: #2B1E1E !important;
}

/* Sembunyikan elemen bawaan Streamlit */
#MainMenu { visibility: hidden !important; display: none !important; }
footer { visibility: hidden !important; display: none !important; }
header { visibility: hidden !important; display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }
#stDecoration { display: none !important; }

/* Pembatasan lebar kontainer desktop fullscreen */
.block-container {
    max-width: 1560px !important;
    width: 96% !important;
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin: 0 auto !important;
}

/* Tabular nums untuk seluruh angka dan persentase */
.tabular-number {
    font-variant-numeric: tabular-nums !important;
    font-feature-settings: "tnum" !important;
}

/* Header Aplikasi Berpenampilan Medis Desktop */
.app-header-box {
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #E6D8CA;
}
.app-header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
}
.app-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.app-emblem {
    width: 48px;
    height: 48px;
    background-color: #9E3B3B;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.app-title {
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.25;
    color: #2B1E1E;
    margin: 0 0 0.2rem 0;
}
.app-subtitle {
    font-size: 0.90rem;
    line-height: 1.45;
    color: #554444;
    margin: 0;
}
.header-stat-pills {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.header-stat-pill {
    background-color: #FFFFFF;
    border: 1px solid #E6D8CA;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #4A3C3C;
    display: flex;
    align-items: center;
    gap: 6px;
}
.header-stat-pill strong {
    color: #9E3B3B;
}

/* Navigasi Desktop Penuh (Segmented Tabs) */
.stRadio > div {
    gap: 0.5rem;
}
.stRadio [role="radiogroup"] {
    background-color: #EDE4D8 !important;
    border: 1px solid #D9CBBE !important;
    padding: 4px 6px !important;
    border-radius: 8px !important;
    display: inline-flex !important;
    flex-direction: row !important;
    gap: 8px !important;
    width: auto !important;
    margin-bottom: 1.5rem !important;
}
.stRadio [role="radiogroup"] label {
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    padding: 7px 18px !important;
    font-size: 0.94rem !important;
    font-weight: 600 !important;
    color: #4A3C3C !important;
    cursor: pointer !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
}
.stRadio [role="radiogroup"] label:hover {
    background-color: #FAF6F2 !important;
    color: #2B1E1E !important;
}
.stRadio [role="radiogroup"] label > div:first-child {
    display: none !important;
}
.stRadio [role="radiogroup"] label[data-checked="true"],
.stRadio [role="radiogroup"] label:has(input:checked) {
    background-color: #FFFFFF !important;
    border: 1px solid #D9CBBE !important;
    color: #9E3B3B !important;
    box-shadow: 0 1px 3px rgba(43, 30, 30, 0.08) !important;
}


/* Panel Desktop Card */
.desktop-panel-card {
    background-color: #FFFFFF;
    border: 1px solid #E6D8CA;
    border-radius: 8px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 1.15rem;
}
.desktop-panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.85rem;
}
.desktop-panel-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #2B1E1E;
    margin: 0;
}

/* Empty State Card untuk Kolom Kanan */
.empty-state-card {
    background-color: #FFFFFF;
    border: 1px dashed #D9CBBE;
    border-radius: 8px;
    padding: 3rem 2rem;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 520px;
}
.empty-state-icon-wrap {
    width: 68px;
    height: 68px;
    border-radius: 14px;
    background-color: #FFEAD3;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1.25rem;
}
.empty-state-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #2B1E1E;
    margin-bottom: 0.5rem;
}
.empty-state-desc {
    font-size: 0.92rem;
    color: #554444;
    max-width: 480px;
    line-height: 1.55;
    margin-bottom: 1.75rem;
}
.empty-features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    width: 100%;
    max-width: 620px;
}
.empty-feature-item {
    background-color: #FAF6F2;
    border: 1px solid #E6D8CA;
    border-radius: 6px;
    padding: 0.85rem 0.65rem;
    text-align: center;
}
.empty-feature-item-title {
    font-size: 0.84rem;
    font-weight: 700;
    color: #2B1E1E;
    margin-bottom: 3px;
}
.empty-feature-item-desc {
    font-size: 0.75rem;
    color: #6E5C5C;
    line-height: 1.35;
    margin: 0;
}

/* Kotak Petunjuk Pemotretan & Kartu Visual */
.instructions-card {
    background-color: #FFEAD3;
    border: 1px solid #9E3B3B;
    border-radius: 8px;
    padding: 1.1rem;
    margin: 1rem 0 1.25rem 0;
}
.instructions-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.5rem;
}
.instructions-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #9E3B3B;
}
.instructions-body {
    font-size: 0.88rem;
    line-height: 1.5;
    color: #2B1E1E;
    margin: 0 0 0.85rem 0;
}

/* Container Cara Memotret (Sesuai Referensi Dua Panel) */
.photo-instructions-container {
    background-color: #FAF6F2;
    border: 1px solid #E6D8CA;
    border-radius: 8px;
    padding: 1.15rem;
    margin: 1rem 0 1.25rem 0;
}
.photo-instructions-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.85rem;
}
.photo-instructions-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #2B1E1E;
    margin: 0;
}
.photo-instructions-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 14px;
}
@media (min-width: 520px) {
    .photo-instructions-grid {
        grid-template-columns: 1fr 1fr;
    }
}
.photo-instructions-card {
    display: flex;
    flex-direction: column;
}
.photo-visual-box {
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 0.55rem;
    border: 1px solid #E6D8CA;
    background-color: #FAF6F2;
}
.photo-caption-correct {
    font-size: 0.84rem;
    line-height: 1.45;
    color: #2B1E1E;
}
.photo-caption-correct strong {
    color: #1F5A37;
    font-weight: 700;
}
.photo-caption-incorrect {
    font-size: 0.84rem;
    line-height: 1.45;
    color: #2B1E1E;
}
.photo-caption-incorrect strong {
    color: #2B1E1E;
    font-weight: 700;
}
.photo-instructions-footer {
    margin-top: 0.85rem;
    padding-top: 0.75rem;
    border-top: 1px solid #E6D8CA;
    font-size: 0.85rem;
    line-height: 1.45;
    color: #554444;
}

/* Galeri Contoh Gambar Luka */
.example-gallery-card {
    background-color: #FFFFFF;
    border: 1px solid #E6D8CA;
    border-radius: 8px;
    padding: 0.85rem;
    margin-bottom: 0.85rem;
}
.example-gallery-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #2B1E1E;
    margin-bottom: 2px;
}
.example-gallery-desc {
    font-size: 0.82rem;
    line-height: 1.45;
    color: #554444;
    margin: 0.35rem 0 0.65rem 0;
}

/* Format Berkas yang Didukung */
.supported-formats-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin: 0.4rem 0 0.8rem 0;
}
.format-pill {
    background-color: #EFE8E0;
    color: #4A3C3C;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid #D9CBBE;
}

/* Area Sentuh Tombol Aksi */
div.stButton > button {
    min-height: 48px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    border: 1px solid #9E3B3B !important;
    background-color: #9E3B3B !important;
    color: #FAF6F2 !important;
    padding: 0.6rem 1.25rem !important;
    box-shadow: none !important;
    transition: none !important;
}
div.stButton > button:hover {
    background-color: #822F2F !important;
    border-color: #822F2F !important;
    color: #FAF6F2 !important;
}

/* Kartu Referensi 5 Kategori Luka */
.category-guide-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
    margin: 0.75rem 0;
}
@media (min-width: 520px) {
    .category-guide-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
.category-card {
    background-color: #FFFFFF;
    border: 1px solid #E6D8CA;
    border-radius: 6px;
    padding: 0.75rem;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.category-card-icon {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.category-card-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #2B1E1E;
    margin-bottom: 2px;
}
.category-card-desc {
    font-size: 0.78rem;
    line-height: 1.35;
    color: #554444;
    margin: 0;
}

/* Hasil Klasifikasi */
.result-header-card {
    background-color: #FFFFFF;
    border: 1px solid #E6D8CA;
    border-radius: 8px;
    padding: 1rem 1.15rem;
    margin: 1rem 0;
}
.result-main-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: #2B1E1E;
    margin: 0 0 0.25rem 0;
}
.result-en-name {
    font-size: 0.92rem;
    color: #6E5C5C;
    margin: 0 0 0.65rem 0;
}

/* Lencana Prioritas */
.priority-badge {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}
.priority-rendah {
    background-color: #E2EDE5;
    color: #1F5A37;
    border: 1px solid #1F5A37;
}
.priority-sedang {
    background-color: #FFF2DF;
    color: #8A5700;
    border: 1px solid #D99B26;
}
.priority-tinggi {
    background-color: #F8E2E2;
    color: #832727;
    border: 1px solid #9E3B3B;
}
.priority-normal {
    background-color: #E2EDE5;
    color: #1F5A37;
    border: 1px solid #3B7A57;
}

/* Komponen Pita Keyakinan */
.confidence-ribbon-container {
    margin: 0.85rem 0 1.25rem 0;
    padding: 0.85rem;
    background-color: #FAF6F2;
    border: 1px solid #E6D8CA;
    border-radius: 6px;
}
.ribbon-track {
    position: relative;
    height: 12px;
    border-radius: 6px;
    display: flex;
    overflow: visible;
    margin: 1.5rem 0 0.6rem 0;
}
.zone-not-detected {
    width: 29%;
    background-color: #526B7A;
    border-top-left-radius: 6px;
    border-bottom-left-radius: 6px;
}
.zone-possible {
    width: 6%;
    background-color: #D99B26;
}
.zone-detected {
    width: 65%;
    background-color: #9E3B3B;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}
.zone-detected-normal {
    width: 65%;
    background-color: #3B7A57;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}

.ribbon-marker {
    position: absolute;
    top: -5px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: #2B1E1E;
    border: 3px solid #FAF6F2;
    transform: translateX(-50%);
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
}
.ribbon-marker-label {
    position: absolute;
    top: -24px;
    transform: translateX(-50%);
    background-color: #2B1E1E;
    color: #FAF6F2;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 3px;
    white-space: nowrap;
}
.ribbon-legend {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #6E5C5C;
    margin-top: 0.35rem;
}

/* Langkah Pertolongan Pertama */
.first-aid-section {
    background-color: #FFFFFF;
    border: 1px solid #E6D8CA;
    border-radius: 8px;
    padding: 1.15rem;
    margin: 1rem 0;
}
.first-aid-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.85rem;
}
.first-aid-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #2B1E1E;
    margin: 0;
}
.first-aid-list {
    margin: 0;
    padding-left: 1.25rem;
}
.first-aid-list li {
    font-size: 0.92rem;
    line-height: 1.55;
    color: #2B1E1E;
    margin-bottom: 0.5rem;
}

/* Kotak Larangan Kritis (Peringatan JANGAN) */
.warning-box {
    background-color: #FFF2DF;
    border-left: 4px solid #9E3B3B;
    border-radius: 0 6px 6px 0;
    padding: 0.85rem 1rem;
    margin: 1rem 0;
}
.warning-box-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.4rem;
}
.warning-box-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #9E3B3B;
    margin: 0;
}
.warning-box-item {
    font-size: 0.88rem;
    line-height: 1.5;
    color: #2B1E1E;
    margin-bottom: 0.35rem;
}

/* Kontak Darurat Medis */
.emergency-contacts-card {
    background-color: #FAF6F2;
    border: 1px solid #D99B26;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin: 1rem 0;
}
.emergency-contacts-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #8A5700;
    margin-bottom: 0.4rem;
}
.emergency-badges-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.emergency-badge {
    background-color: #FFFFFF;
    border: 1px solid #E6D8CA;
    border-radius: 6px;
    padding: 0.35rem 0.65rem;
    font-size: 0.82rem;
    color: #2B1E1E;
}
.emergency-badge strong {
    color: #9E3B3B;
}

/* Expander Rekayasa Penalaran */
div[data-testid="stExpander"] {
    background-color: #FAF6F2 !important;
    border: 1px solid #E6D8CA !important;
    border-radius: 6px !important;
    margin: 1rem 0 !important;
}
div[data-testid="stExpander"] details summary {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #2B1E1E !important;
}

/* Penafian Medis & Footer */
.footer-disclaimer-box {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #E6D8CA;
    font-size: 0.8rem;
    line-height: 1.5;
    color: #6E5C5C;
}
.footer-disclaimer-box a {
    color: #9E3B3B;
    text-decoration: underline;
}

/* Komponen Halaman Tentang & Metrik */
.about-card {
    background-color: #FFFFFF;
    border: 1px solid #E6D8CA;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.about-card-title {
    font-size: 1rem;
    font-weight: 700;
    color: #9E3B3B;
    margin-bottom: 0.45rem;
}
.about-card-body {
    font-size: 0.9rem;
    line-height: 1.55;
    color: #2B1E1E;
    margin: 0;
}

/* Grid & Kartu Metrik Evaluasi Model Desktop */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin: 0.75rem 0 1.35rem 0;
}
@media (max-width: 1100px) {
    .metric-grid {
        grid-template-columns: repeat(3, 1fr);
    }
}
@media (max-width: 600px) {
    .metric-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E6D8CA;
    border-radius: 8px;
    padding: 0.85rem;
    text-align: center;
}
.metric-card-val {
    font-size: 1.45rem;
    font-weight: 700;
    color: #9E3B3B;
    margin-bottom: 2px;
}
.metric-card-lbl {
    font-size: 0.78rem;
    font-weight: 600;
    color: #2B1E1E;
}
.metric-card-sub {
    font-size: 0.72rem;
    color: #6E5C5C;
    margin-top: 2px;
}

/* Tabel Metrik Performa */
.metric-table-container {
    overflow-x: auto;
    margin: 0.75rem 0 1.25rem 0;
    border: 1px solid #E6D8CA;
    border-radius: 8px;
    background-color: #FFFFFF;
}
.metric-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.metric-table th {
    background-color: #FFEAD3;
    color: #2B1E1E;
    font-weight: 700;
    padding: 9px 12px;
    border-bottom: 1px solid #E6D8CA;
    text-align: left;
    white-space: nowrap;
}
.metric-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #F2E8DC;
    color: #2B1E1E;
    white-space: nowrap;
}
.metric-table tr:last-child td {
    border-bottom: none;
}
.metric-table tr:nth-child(even) td {
    background-color: #FCF9F6;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Komponen Visual: Pita Keyakinan (Three-zone Horizontal Bar)
# -----------------------------------------------------------------------------
def render_confidence_ribbon(confidence: float, status: str, is_normal: bool = False):
    percent = round(confidence * 100, 1)
    clamped_pos = max(2.0, min(98.0, percent))
    zone3_class = "zone-detected-normal" if is_normal else "zone-detected"

    html = f"""
    <div class="confidence-ribbon-container">
        <div style="font-size: 0.84rem; font-weight: 600; color: #2B1E1E; margin-bottom: 0.2rem;">
            Tingkat keyakinan deteksi
        </div>
        <div class="ribbon-track">
            <div class="zone-not-detected" title="Tidak terdeteksi (< 29%)"></div>
            <div class="zone-possible" title="Kemungkinan (29% - 35%)"></div>
            <div class="{zone3_class}" title="Terdeteksi (>= 35%)"></div>
            <div class="ribbon-marker" style="left: {clamped_pos}%;">
                <div class="ribbon-marker-label tabular-number">{percent}%</div>
            </div>
        </div>
        <div class="ribbon-legend">
            <span>Tidak terdeteksi (&lt;29%)</span>
            <span>Kemungkinan (29-35%)</span>
            <span>Terdeteksi (&ge;35%)</span>
        </div>
    </div>
    """
    safe_markdown(html)

# -----------------------------------------------------------------------------
# Header Aplikasi dengan Lambang Medis Resmi
# -----------------------------------------------------------------------------
header_html = f"""
<div class="app-header-box">
    <div class="app-header-inner">
        <div class="app-header-left">
            <div class="app-emblem" aria-hidden="true">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="10" y="4" width="4" height="16" rx="1.5" fill="#FAF6F2"/>
                    <rect x="4" y="10" width="16" height="4" rx="1.5" fill="#FAF6F2"/>
                </svg>
            </div>
            <div>
                <h1 class="app-title">{APP_CONFIG.get('app', {}).get('title', 'Deteksi Luka Kulit & Rekomendasi P3K')}</h1>
                <p class="app-subtitle">{APP_CONFIG.get('app', {}).get('subtitle', 'Sistem panduan pertolongan pertama awal berbasis deteksi visual dan penalaran aturan medis')}</p>
            </div>
        </div>
        <div class="header-stat-pills">
            <div class="header-stat-pill">Model ONNX: <strong>YOLOv8 Nano</strong></div>
            <div class="header-stat-pill">Akurasi: <strong>86.4% mAP@0.5</strong></div>
            <div class="header-stat-pill">Mesin Aturan: <strong>R001–R005</strong></div>
        </div>
    </div>
</div>
"""
safe_markdown(header_html)

# Navigasi halaman
current_page = st.radio(
    label="Pilih menu navigasi",
    options=["Pemeriksaan luka", "Metrik evaluasi model", "Tentang sistem"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

# -----------------------------------------------------------------------------
# Halaman 1: Pemeriksaan Luka (Alur Utama)
# -----------------------------------------------------------------------------
if current_page == "Pemeriksaan luka":
    # Inisialisasi session state untuk pemilihan contoh citra
    if "chosen_example_path" not in st.session_state:
        st.session_state["chosen_example_path"] = None

    col_workspace, col_analysis = st.columns([1, 1.25], gap="large")

    with col_workspace:
        # 1. Card Masukan Foto Kulit
        safe_markdown("""
        <div class="desktop-panel-card">
            <div class="desktop-panel-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2B1E1E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                <div class="desktop-panel-title">Masukan foto kulit</div>
            </div>
        </div>
        """)

        input_method = st.radio(
            label="Metode pengambilan foto",
            options=["Unggah foto", "Ambil foto luka"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )

        raw_image = None
        if input_method == "Ambil foto luka":
            camera_file = st.camera_input("Arahkan kamera ke area luka kulit")
            if camera_file is not None:
                st.session_state["chosen_example_path"] = None
                raw_image = camera_file
        else:
            allowed_exts = APP_CONFIG.get("app", {}).get(
                "allowed_extensions",
                ["jpg", "jpeg", "png", "webp", "avif", "bmp", "tiff", "heic"]
            )
            safe_markdown("""
            <div class="supported-formats-bar">
                <span style="font-size: 0.78rem; font-weight: 600; color: #6E5C5C;">Format didukung:</span>
                <span class="format-pill">JPG</span>
                <span class="format-pill">PNG</span>
                <span class="format-pill">WEBP</span>
                <span class="format-pill">AVIF</span>
                <span class="format-pill">BMP</span>
                <span class="format-pill">TIFF</span>
                <span class="format-pill">HEIC</span>
                <span style="font-size: 0.76rem; color: #6E5C5C; margin-left: 4px;">(maks. 10 MB)</span>
            </div>
            """)
            uploaded_file = st.file_uploader(
                label="Pilih berkas foto kulit",
                type=allowed_exts,
                accept_multiple_files=False,
                help="Mendukung foto JPEG, PNG, WEBP, AVIF, BMP, TIFF, atau HEIC dengan ukuran maksimal 10 MB"
            )
            if uploaded_file is not None:
                max_size_bytes = APP_CONFIG.get("app", {}).get("max_upload_size_mb", 10) * 1024 * 1024
                if uploaded_file.size > max_size_bytes:
                    st.error("Ukuran berkas melebihi batas maksimal 10 MB. Silakan unggah foto dengan resolusi lebih ringkas.")
                else:
                    st.session_state["chosen_example_path"] = None
                    raw_image = uploaded_file

        # 2. Card Panduan Cara Memotret (Dual Panel SVG)
        safe_markdown("""
        <div class="desktop-panel-card" style="margin-top: 1rem;">
            <div class="desktop-panel-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2B1E1E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                    <circle cx="12" cy="13" r="4"></circle>
                </svg>
                <div class="desktop-panel-title">Panduan cara memotret</div>
            </div>
        </div>
        """)

        col_cam_benar, col_cam_salah = st.columns(2)
        img_b_path = os.path.join(CURRENT_DIR, "static", "cara_memotret_benar.svg")
        if not os.path.exists(img_b_path):
            img_b_path = os.path.join(PARENT_DIR, "static", "cara_memotret_benar.svg")
        if not os.path.exists(img_b_path):
            img_b_path = "app/static/cara_memotret_benar.svg"

        img_s_path = os.path.join(CURRENT_DIR, "static", "cara_memotret_salah.svg")
        if not os.path.exists(img_s_path):
            img_s_path = os.path.join(PARENT_DIR, "static", "cara_memotret_salah.svg")
        if not os.path.exists(img_s_path):
            img_s_path = "app/static/cara_memotret_salah.svg"

        with col_cam_benar:
            st.image(img_b_path, use_container_width=True)
            safe_markdown('<div class="photo-caption-correct"><strong>✓ Seperti ini.</strong> Fokus dekat tajam, area luka memenuhi sebagian besar foto.</div>')
        with col_cam_salah:
            st.image(img_s_path, use_container_width=True)
            safe_markdown('<div class="photo-caption-incorrect"><strong>✕ Bukan seperti ini.</strong> Terlalu jauh, buram, sudut miring ekstrem, atau gelap.</div>')

        safe_markdown('<div class="photo-instructions-footer">Foto dari jarak sekitar 10–20 cm dengan cahaya cukup terang tanpa pantulan silau.</div>')

        # 3. Card Uji Cepat Sampel Citra Penelitian
        safe_markdown("""
        <div class="desktop-panel-card" style="margin-top: 1rem;">
            <div class="desktop-panel-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2B1E1E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                    <polyline points="2 17 12 22 22 17"></polyline>
                    <polyline points="2 12 12 17 22 12"></polyline>
                </svg>
                <div class="desktop-panel-title">Uji cepat dengan citra sampel riil</div>
            </div>
            <p style="font-size: 0.82rem; color: #554444; margin-bottom: 0.75rem;">
                Pilih salah satu sampel riil dari dataset 1.391 citra untuk menguji respons sistem:
            </p>
        </div>
        """)

        tab_abr, tab_bru, tab_bur, tab_cut, tab_nor = st.tabs([
            "Luka lecet",
            "Luka memar",
            "Luka bakar",
            "Luka sayat",
            "Kulit normal"
        ])
        with tab_abr:
            col_img, col_desc = st.columns([1, 1.25])
            with col_img:
                st.image(resolve_sample_path("sample_images/Abrasions_5.jpg"), caption="Contoh citra Luka lecet (Abrasion)", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div style="font-size: 0.95rem; font-weight: 700; color: #2B1E1E;">Luka lecet (Abrasion)</div>
                <span class="priority-badge priority-rendah">Prioritas Rendah</span>
                <p style="font-size: 0.8rem; line-height: 1.4; color: #4A3C3C; margin: 0.35rem 0;">
                    Epidermis tergores atau terkelupas, kemerahan merata di permukaan akibat gesekan mekanik.
                </p>
                """)
                if st.button("Uji deteksi foto ini", key="btn_sample_abr"):
                    st.session_state["chosen_example_path"] = resolve_sample_path("sample_images/Abrasions_5.jpg")
                    st.rerun()

        with tab_bru:
            col_img, col_desc = st.columns([1, 1.25])
            with col_img:
                st.image(resolve_sample_path("sample_images/Bruise_11.jpg"), caption="Contoh citra Luka memar (Bruise)", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div style="font-size: 0.95rem; font-weight: 700; color: #2B1E1E;">Luka memar (Bruise)</div>
                <span class="priority-badge priority-rendah">Prioritas Rendah</span>
                <p style="font-size: 0.8rem; line-height: 1.4; color: #4A3C3C; margin: 0.35rem 0;">
                    Perubahan warna jaringan subkutan (kebiruan/keunguan) akibat pecahnya kapiler tanpa luka terbuka.
                </p>
                """)
                if st.button("Uji deteksi foto ini", key="btn_sample_bru"):
                    st.session_state["chosen_example_path"] = resolve_sample_path("sample_images/Bruise_11.jpg")
                    st.rerun()

        with tab_bur:
            col_img, col_desc = st.columns([1, 1.25])
            with col_img:
                st.image(resolve_sample_path("sample_images/Burn_1.jpg"), caption="Contoh citra Luka bakar (Burn)", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div style="font-size: 0.95rem; font-weight: 700; color: #2B1E1E;">Luka bakar (Burn)</div>
                <span class="priority-badge priority-sedang">Prioritas Sedang / Tinggi</span>
                <p style="font-size: 0.8rem; line-height: 1.4; color: #4A3C3C; margin: 0.35rem 0;">
                    Eritema meluas, jaringan melepuh atau timbul bula berisi cairan akibat panas atau api.
                </p>
                """)
                if st.button("Uji deteksi foto ini", key="btn_sample_bur"):
                    st.session_state["chosen_example_path"] = resolve_sample_path("sample_images/Burn_1.jpg")
                    st.rerun()

        with tab_cut:
            col_img, col_desc = st.columns([1, 1.25])
            with col_img:
                st.image(resolve_sample_path("sample_images/Cut_4.jpg"), caption="Contoh citra Luka sayat (Cut)", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div style="font-size: 0.95rem; font-weight: 700; color: #2B1E1E;">Luka sayat (Cut)</div>
                <span class="priority-badge priority-sedang">Prioritas Sedang / Tinggi</span>
                <p style="font-size: 0.8rem; line-height: 1.4; color: #4A3C3C; margin: 0.35rem 0;">
                    Garis sayatan atau robekan linier bertepi tegas akibat benda tajam dengan resiko perdarahan aktif.
                </p>
                """)
                if st.button("Uji deteksi foto ini", key="btn_sample_cut"):
                    st.session_state["chosen_example_path"] = resolve_sample_path("sample_images/Cut_4.jpg")
                    st.rerun()

        with tab_nor:
            col_img, col_desc = st.columns([1, 1.25])
            with col_img:
                st.image(resolve_sample_path("sample_images/no_abnormality_2.jpg"), caption="Contoh citra Kulit normal (No Abnormality)", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div style="font-size: 0.95rem; font-weight: 700; color: #2B1E1E;">Kulit normal (No Abnormality)</div>
                <span class="priority-badge priority-normal">Prioritas Normal</span>
                <p style="font-size: 0.8rem; line-height: 1.4; color: #4A3C3C; margin: 0.35rem 0;">
                    Sawar kulit utuh dan sehat alami tanpa lesi patologis, pembengkakan, maupun perdarahan.
                </p>
                """)
                if st.button("Uji deteksi foto ini", key="btn_sample_nor"):
                    st.session_state["chosen_example_path"] = resolve_sample_path("sample_images/no_abnormality_2.jpg")
                    st.rerun()

    # Periksa apakah ada citra yang dipilih dari galeri sampel
    if raw_image is None and st.session_state.get("chosen_example_path"):
        ex_path = st.session_state["chosen_example_path"]
        if os.path.exists(ex_path):
            with open(ex_path, "rb") as sf:
                raw_image = BytesIO(sf.read())

    with col_analysis:
        if raw_image is None:
            safe_markdown("""
            <div class="empty-state-card">
                <div class="empty-state-icon-wrap">
                    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                    </svg>
                </div>
                <div class="empty-state-title">Ruang Analisis &amp; Rekomendasi Medis</div>
                <div class="empty-state-desc">
                    Silakan unggah foto luka kulit atau pilih salah satu sampel citra dari panel sebelah kiri. Sistem akan langsung mendeteksi lesi dan menyimpulkan pertolongan pertama berbasis aturan medis.
                </div>
                <div class="empty-features-grid">
                    <div class="empty-feature-item">
                        <div class="empty-feature-item-title">Deteksi YOLOv8</div>
                        <div class="empty-feature-item-desc">5 Kategori lesi kulit dengan presisi tinggi</div>
                    </div>
                    <div class="empty-feature-item">
                        <div class="empty-feature-item-title">Pita Keyakinan</div>
                        <div class="empty-feature-item-desc">3 Zona ambang batas (&lt;29%, 29-35%, &ge;35%)</div>
                    </div>
                    <div class="empty-feature-item">
                        <div class="empty-feature-item-title">Aturan P3K Medis</div>
                        <div class="empty-feature-item-desc">Penalaran Forward Chaining R001–R005</div>
                    </div>
                </div>
            </div>
            """)
        else:
            try:
                pil_image = Image.open(raw_image)
                pil_image = pil_image.convert("RGB")
            except Exception as e:
                st.error("Format berkas tidak valid, tidak didukung, atau rusak. Pastikan berkas adalah citra yang valid (JPG, PNG, WEBP, AVIF, BMP, atau TIFF).")
                st.stop()

            detector = get_detector()
            engine = get_rule_engine()

            with st.spinner("Memproses foto dan menganalisis kondisi kulit..."):
                detections, annotated_rgb = detector.predict(pil_image)

            top_detection = detections[0] if detections else {
                "class_name": "No Abnormality",
                "label_id": "Kulit normal",
                "label_en": "No Abnormality",
                "confidence": 0.0,
                "status": "not_detected",
                "bbox": [0, 0, 0, 0],
                "area_ratio": 0.0
            }

            triage = engine.evaluate(top_detection, top_detection.get("status", "detected"))

            st.markdown("<h2 style='font-size: 1.15rem; font-weight: 700; margin: 0 0 0.5rem 0;'>Hasil analisis foto</h2>", unsafe_allow_html=True)

            st.image(
                annotated_rgb,
                caption="Tampilan area lesi dengan kotak pembatas deteksi YOLOv8",
                use_container_width=True
            )

            label_id = triage.get("label_id", top_detection.get("label_id", "Kulit normal"))
            label_en = triage.get("label_en", top_detection.get("label_en", "No Abnormality"))
            priority = triage.get("priority", "Rendah")
            confidence = float(top_detection.get("confidence", 0.0))
            status = top_detection.get("status", "detected")
            is_normal = "normal" in label_en.lower() or "abnormality" in label_en.lower()

            priority_class = "priority-rendah"
            if priority.lower() == "sedang":
                priority_class = "priority-sedang"
            elif priority.lower() in ["tinggi", "gawat darurat"]:
                priority_class = "priority-tinggi"
            elif is_normal or priority.lower() == "normal":
                priority_class = "priority-normal"

            result_header_html = f"""
            <div class="result-header-card">
                <h3 class="result-main-name">{label_id}</h3>
                <p class="result-en-name">{label_en}</p>
                <span class="priority-badge {priority_class}">Tingkat prioritas: {priority}</span>
            </div>
            """
            safe_markdown(result_header_html)

            # Komponen Visual Khas: Pita Keyakinan Tiga Zona
            render_confidence_ribbon(confidence, status, is_normal=is_normal)

            # Kontak Darurat Medis (Tampil bila luka berprioritas sedang atau tinggi)
            if priority.lower() in ["sedang", "tinggi", "gawat darurat"]:
                safe_markdown(
                    """
                    <div class="emergency-contacts-card">
                        <div class="emergency-contacts-title">Layanan darurat medis cepat (Indonesia)</div>
                        <div class="emergency-badges-wrap">
                            <span class="emergency-badge">Ambulans: <strong>119</strong></span>
                            <span class="emergency-badge">Panggilan Darurat: <strong>112</strong></span>
                            <span class="emergency-badge">PMI: <strong>021-7992325</strong></span>
                        </div>
                    </div>
                    """
                )

            # Langkah Pertolongan Pertama (First Aid Steps)
            actions = triage.get("actions", [])
            if actions:
                action_items_html = "".join([f"<li>{act}</li>" for act in actions])
                first_aid_html = f"""
                <div class="first-aid-section">
                    <div class="first-aid-header">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2B1E1E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <div class="first-aid-title">Langkah pertolongan pertama</div>
                    </div>
                    <ol class="first-aid-list">
                        {action_items_html}
                    </ol>
                </div>
                """
                safe_markdown(first_aid_html)

            # Penekanan Visual Larangan Kritis (Peringatan JANGAN)
            critical_warnings = triage.get("critical_warnings", [])
            if critical_warnings:
                warning_items_html = "".join([f"<div class='warning-box-item'>{w}</div>" for w in critical_warnings])
                warning_html = f"""
                <div class="warning-box">
                    <div class="warning-box-header">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                            <line x1="12" y1="9" x2="12" y2="13"></line>
                            <line x1="12" y1="17" x2="12.01" y2="17"></line>
                        </svg>
                        <div class="warning-box-title">Perhatian kritis (larangan penting)</div>
                    </div>
                    {warning_items_html}
                </div>
                """
                safe_markdown(warning_html)

            # Jejak Penalaran Forward Chaining (st.expander terbuka-tutup)
            with st.expander("Mengapa sistem menyimpulkan ini"):
                rule_id = triage.get("rule_id", "R005")
                st.markdown(f"**ID Aturan Pakar yang Aktif:** `{rule_id}`")
                st.markdown("**Rantai Fakta Masukan:**")
                for step in triage.get("reasoning_steps", []):
                    st.markdown(f"- {step}")
                st.markdown(f"**Kesimpulan Inferensi:** {triage.get('conclusion', '')}")
                if triage.get("notes"):
                    st.markdown(f"**Catatan Medis Tambahan:** {triage.get('notes')}")

            # Tombol Reset / Analisis Baru
            if st.button("Pemeriksaan foto lain", key="btn_reset_analysis"):
                st.session_state["chosen_example_path"] = None
                st.rerun()

    # Catatan Kaki & Penafian Singkat yang Tenang (Spanning full width)
    safe_markdown(
        """
        <div class="footer-disclaimer-box">
            Sistem ini hanya mendeteksi 5 kategori permukaan kulit (luka lecet, luka memar, luka bakar, luka sayat, dan kulit normal).
            Hasil pemeriksaan berupa panduan pertolongan pertama awal dan bukan pengganti diagnosis tenaga medis profesional.
            Pelajari metode kerja dan batas kemampuan sistem di halaman <a href="?nav=about">Tentang sistem</a>.
        </div>
        """
    )

# -----------------------------------------------------------------------------
# Halaman 2: Metrik Evaluasi Model (Kinerja Riset YOLOv8)
# -----------------------------------------------------------------------------
elif current_page == "Metrik evaluasi model":
    st.markdown("<h2 style='font-size: 1.25rem; font-weight: 700; margin: 0.5rem 0 0.35rem 0;'>Metrik evaluasi model</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.88rem; color: #554444; margin-bottom: 1.25rem;'>Kinerja kuantitatif model YOLOv8 pada External Wound Dataset (1.391 citra) sesuai Subbab 3.5 Proposal Tugas Akhir</p>", unsafe_allow_html=True)

    metrics_ov = BENCHMARK_RESULTS.get("metrics_overall", {})
    metrics_cls = BENCHMARK_RESULTS.get("metrics_per_class", {})
    dataset_info = BENCHMARK_RESULTS.get("dataset_info", {})
    conf_mat = BENCHMARK_RESULTS.get("confusion_matrix", {})

    # 1. Kartu Ringkasan Metrik Utama (6 Parameter)
    safe_markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-card-val tabular-number">{metrics_ov.get('mAP_50', 0.864) * 100:.1f}%</div>
                <div class="metric-card-lbl">mAP@0.5</div>
                <div class="metric-card-sub">IoU Ambang 0.50</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-val tabular-number">{metrics_ov.get('mAP_50_95', 0.612) * 100:.1f}%</div>
                <div class="metric-card-lbl">mAP@0.5:0.95</div>
                <div class="metric-card-sub">Rerata IoU Multi-ambang</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-val tabular-number">{metrics_ov.get('precision', 0.852) * 100:.1f}%</div>
                <div class="metric-card-lbl">Presisi (Precision)</div>
                <div class="metric-card-sub">Ketepatan deteksi positif</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-val tabular-number">{metrics_ov.get('recall', 0.838) * 100:.1f}%</div>
                <div class="metric-card-lbl">Sensitivitas (Recall)</div>
                <div class="metric-card-sub">Cakupan lesi terdeteksi</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-val tabular-number">{metrics_ov.get('f1_score', 0.845) * 100:.1f}%</div>
                <div class="metric-card-lbl">F1-Score</div>
                <div class="metric-card-sub">Keseimbangan P &amp; R</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-val tabular-number">{metrics_ov.get('inference_speed_ms', 14.8)} ms</div>
                <div class="metric-card-lbl">Kecepatan Inferensi</div>
                <div class="metric-card-sub">Setara ~{metrics_ov.get('fps', 67.5)} FPS</div>
            </div>
        </div>
        """
    )

    # 2. Layout Desktop Dua Kolom untuk Evaluasi Mendalam
    col_p2_left, col_p2_right = st.columns([1.15, 0.85], gap="large")

    with col_p2_left:
        # Tabel Evaluasi Per Kelas
        st.markdown("<h3 style='font-size: 1.05rem; font-weight: 700; color: #2B1E1E; margin: 0 0 0.5rem 0;'>Kinerja deteksi per kategori luka</h3>", unsafe_allow_html=True)
        table_rows = ""
        for raw_c, data in metrics_cls.items():
            lbl_id = data.get("label_id", raw_c)
            table_rows += f"""
            <tr>
                <td><strong>{lbl_id}</strong> <span style="color: #6E5C5C;">({raw_c})</span></td>
                <td class="tabular-number">{data.get('precision', 0) * 100:.1f}%</td>
                <td class="tabular-number">{data.get('recall', 0) * 100:.1f}%</td>
                <td class="tabular-number">{data.get('f1_score', 0) * 100:.1f}%</td>
                <td class="tabular-number" style="font-weight: 700; color: #9E3B3B;">{data.get('mAP_50', 0) * 100:.1f}%</td>
                <td class="tabular-number">{data.get('instances', 0)}</td>
            </tr>
            """

        safe_markdown(
            f"""
            <div class="metric-table-container">
                <table class="metric-table">
                    <thead>
                        <tr>
                            <th>Kategori Kondisi Kulit</th>
                            <th>Presisi</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>mAP@0.5</th>
                            <th>Sampel Uji</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            """
        )

        # Formulasi Matematis Metrik (Bab 3.5 Proposal)
        with st.expander("Formula evaluasi matematis (Bab 3.5 Proposal)"):
            st.markdown(
                textwrap.dedent(
                    """
                    Metrik performa dihitung berdasarkan formulasi matematis baku deteksi objek:
                    - **Intersection over Union (IoU - Persamaan 3.1)**:
                      $$IoU = \\frac{\\text{Area}(A \\cap B)}{\\text{Area}(A \\cup B)}$$
                    - **Presisi (Precision - Persamaan 3.5)**:
                      $$\\text{Precision} = \\frac{TP}{TP + FP}$$
                    - **Sensitivitas (Recall - Persamaan 3.6)**:
                      $$\\text{Recall} = \\frac{TP}{TP + FN}$$
                    - **F1-Score (Persamaan 3.7)**:
                      $$F1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$
                    - **Mean Average Precision (mAP@0.5)**:
                      $$\\text{mAP} = \\frac{1}{N} \\sum_{i=1}^{N} \\text{AP}_i$$
                    di mana $N = 5$ kelas kategori lesi kulit.
                    """
                ).strip()
            )

    with col_p2_right:
        # Matriks Konfusi (Confusion Matrix)
        st.markdown("<h3 style='font-size: 1.05rem; font-weight: 700; color: #2B1E1E; margin: 0 0 0.5rem 0;'>Matriks konfusi data uji</h3>", unsafe_allow_html=True)
        cm_labels = conf_mat.get("labels", [])
        cm_matrix = conf_mat.get("matrix", [])

        cm_th = "".join([f"<th>{l}</th>" for l in cm_labels])
        cm_rows = ""
        for i, row in enumerate(cm_matrix):
            actual_label = cm_labels[i] if i < len(cm_labels) else f"Kelas {i}"
            cells = ""
            for j, val in enumerate(row):
                is_diagonal = (i == j)
                bg_style = ' style="background-color: #E2EDE5; font-weight: 700; color: #1F5A37;"' if is_diagonal else ''
                cells += f'<td class="tabular-number"{bg_style}>{val}</td>'
            cm_rows += f"<tr><td><strong>{actual_label}</strong></td>{cells}</tr>"

        safe_markdown(
            f"""
            <div class="metric-table-container">
                <table class="metric-table">
                    <thead>
                        <tr>
                            <th>Aktual \\ Prediksi</th>
                            {cm_th}
                        </tr>
                    </thead>
                    <tbody>
                        {cm_rows}
                    </tbody>
                </table>
            </div>
            <p style="font-size: 0.76rem; color: #6E5C5C; margin: -0.5rem 0 1rem 0;">Kolom diagonal hijau menunjukkan jumlah prediksi yang diklasifikasikan dengan benar ke kelas aslinya.</p>
            """
        )

        # Distribusi Dataset Penelitian
        splits = dataset_info.get("splits", {})
        safe_markdown(
            f"""
            <div class="about-card" style="margin-top: 0;">
                <div class="about-card-title">Distribusi dataset penelitian (1.391 citra)</div>
                <p class="about-card-body">
                    Model dilatih menggunakan dataset gabungan <strong>External Wound Dataset</strong> (Roboflow Universe &amp; Kaggle) yang dibagi secara proporsional:
                </p>
                <div style="display: flex; gap: 10px; margin-top: 0.75rem; flex-wrap: wrap;">
                    <div style="background-color: #FAF6F2; border: 1px solid #E6D8CA; border-radius: 6px; padding: 0.6rem 0.85rem; flex: 1; min-width: 105px;">
                        <div style="font-size: 0.76rem; color: #6E5C5C;">Data Latih (Train)</div>
                        <div class="tabular-number" style="font-size: 1.15rem; font-weight: 700; color: #2B1E1E;">{splits.get('train', 979)} <span style="font-size: 0.82rem; font-weight: 500; color: #6E5C5C;">(70%)</span></div>
                    </div>
                    <div style="background-color: #FAF6F2; border: 1px solid #E6D8CA; border-radius: 6px; padding: 0.6rem 0.85rem; flex: 1; min-width: 105px;">
                        <div style="font-size: 0.76rem; color: #6E5C5C;">Data Validasi (Valid)</div>
                        <div class="tabular-number" style="font-size: 1.15rem; font-weight: 700; color: #2B1E1E;">{splits.get('valid', 275)} <span style="font-size: 0.82rem; font-weight: 500; color: #6E5C5C;">(20%)</span></div>
                    </div>
                    <div style="background-color: #FAF6F2; border: 1px solid #E6D8CA; border-radius: 6px; padding: 0.6rem 0.85rem; flex: 1; min-width: 105px;">
                        <div style="font-size: 0.76rem; color: #6E5C5C;">Data Uji (Test)</div>
                        <div class="tabular-number" style="font-size: 1.15rem; font-weight: 700; color: #2B1E1E;">{splits.get('test', 137)} <span style="font-size: 0.82rem; font-weight: 500; color: #6E5C5C;">(10%)</span></div>
                    </div>
                </div>
            </div>
            """
        )

        # Panduan Menjalankan Evaluasi Mandiri di Terminal
        with st.expander("Cara mengecek metrik mandiri via terminal (CLI)"):
            st.markdown(
                textwrap.dedent(
                    """
                    Untuk menghitung ulang metrik evaluasi secara langsung menggunakan berkas bobot model PyTorch (`wound_yolov8n.pt`) dan split data uji, jalankan perintah berikut di PowerShell atau terminal:
                    ```powershell
                    python scripts/evaluate.py --model models/wound_yolov8n.pt --data dataset/data.yaml --split test
                    ```
                    Skrip tersebut akan memvalidasi seluruh 137 citra uji dan menghasilkan laporan ringkasan mAP@0.5, mAP@0.5:0.95, Precision, Recall, serta F1-Score di konsol.
                    """
                ).strip()
            )

# -----------------------------------------------------------------------------
# Halaman 3: Tentang Sistem & Atribusi
# -----------------------------------------------------------------------------
elif current_page == "Tentang sistem":
    st.markdown("<h2 style='font-size: 1.25rem; font-weight: 700; margin: 0.5rem 0 0.35rem 0;'>Tentang sistem</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.88rem; color: #554444; margin-bottom: 1.25rem;'>Informasi arsitektur, atribusi dataset, privasi, serta batas kemampuan sistem deteksi luka dan penalaran P3K</p>", unsafe_allow_html=True)

    col_about_1, col_about_2 = st.columns(2, gap="large")

    with col_about_1:
        # Cara Kerja Singkat
        safe_markdown(
            f"""
            <div class="about-card">
                <div class="about-card-title">Arsitektur &amp; cara kerja sistem</div>
                <p class="about-card-body">
                    {APP_CONFIG.get('about', {}).get('system_description',
                    'Sistem ini mengintegrasikan model deteksi visual berbasis YOLOv8 (ONNX) untuk melokalisasi area luka dan menghitung skor keyakinan, dengan mesin penalaran terstruktur Forward Chaining (aturan R001-R005) guna menyimpulkan tingkat keparahan serta urutan langkah pertolongan pertama.')}
                </p>
            </div>
            """
        )

        # Sumber Data & Atribusi
        safe_markdown(
            f"""
            <div class="about-card">
                <div class="about-card-title">Sumber data dan atribusi</div>
                <p class="about-card-body">
                    {APP_CONFIG.get('about', {}).get('dataset_attribution',
                    'Dataset penelitian bersumber dari External Wound Dataset (Roboflow Universe & Kaggle) berjumlah 1.391 citra lesi kulit yang dianotasi ke dalam 5 kelas.')}
                </p>
            </div>
            """
        )

    with col_about_2:
        # Pernyataan Privasi
        safe_markdown(
            f"""
            <div class="about-card">
                <div class="about-card-title">Pernyataan privasi pengguna</div>
                <p class="about-card-body">
                    {APP_CONFIG.get('about', {}).get('privacy_statement',
                    'Foto yang Anda ambil atau unggah diproses secara langsung di memori lokal saat sesi berjalan. Sistem tidak menyimpan citra di penyimpanan permanen atau basis data eksternal, dan foto langsung dibersihkan setelah proses analisis.')}
                </p>
            </div>
            """
        )

        # Peringatan Medis Lengkap
        safe_markdown(
            f"""
            <div class="about-card" style="border-left: 4px solid #9E3B3B;">
                <div class="about-card-title" style="color: #9E3B3B;">Peringatan medis resmi</div>
                <p class="about-card-body">
                    {APP_CONFIG.get('about', {}).get('medical_disclaimer',
                    'Sistem hanya mendeteksi 5 kategori permukaan kulit. Akurasi deteksi dipengaruhi oleh pencahayaan dan sudut pemotretan. Seluruh hasil berupa panduan pertolongan pertama awal (P3K) dan bukan pengganti diagnosis dokter atau tenaga medis profesional.')}
                </p>
            </div>
            """
        )

    safe_markdown(
        """
        <div class="footer-disclaimer-box">
            Aplikasi ini dikembangkan untuk keperluan penelitian akademis dan edukasi pertolongan pertama.
        </div>
        """
    )

