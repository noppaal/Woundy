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
# Injeksi Desain Sistem CSS Modern & Elegan (Clinical-Grade Aesthetics)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

/* Reset dan Aturan Tipografi Global */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #FAF6F2 !important;
    color: #2B1E1E !important;
    -webkit-font-smoothing: antialiased;
}

/* Sembunyikan elemen bawaan Streamlit secara bersih */
#MainMenu { visibility: hidden !important; display: none !important; }
footer { visibility: hidden !important; display: none !important; }
header { visibility: hidden !important; display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }
#stDecoration { display: none !important; }

/* Custom Scrollbar yang Tenang */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #FAF6F2;
}
::-webkit-scrollbar-thumb {
    background: #D9CBBE;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #BFAEA0;
}

/* Pembatasan lebar kontainer desktop fullscreen */
.block-container {
    max-width: 1560px !important;
    width: 96% !important;
    padding-top: 1.25rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin: 0 auto !important;
}

/* Tabular nums untuk seluruh angka dan persentase */
.tabular-number {
    font-variant-numeric: tabular-nums !important;
    font-feature-settings: "tnum" !important;
}

/* ========================================================================= */
/* HEADER DESKTOP ELEGAN (Clinical Medical Header)                           */
/* ========================================================================= */
.app-header-card {
    background: #FFFFFF;
    border: 1px solid #E8DDD1;
    border-radius: 14px;
    padding: 1.25rem 1.6rem;
    margin-bottom: 1.35rem;
    box-shadow: 0 3px 14px rgba(43, 30, 30, 0.04), 0 1px 3px rgba(43, 30, 30, 0.02);
}
.app-header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
}
.app-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.app-emblem {
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #9E3B3B 0%, #852C2C 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(158, 59, 59, 0.28);
    position: relative;
}
.app-emblem::after {
    content: '';
    position: absolute;
    width: 100%;
    height: 100%;
    border-radius: 12px;
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.35);
}
.app-title-wrap {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.app-title-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.app-title {
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1.2;
    color: #2B1E1E;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-badge-version {
    background-color: #FFEAD3;
    color: #8C4700;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    border: 1px solid #F5D3B0;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.app-subtitle {
    font-size: 0.88rem;
    line-height: 1.45;
    color: #635252;
    margin: 0;
}
.header-stat-pills {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.header-stat-pill {
    background-color: #FAF6F2;
    border: 1px solid #E8DDD1;
    border-radius: 8px;
    padding: 0.45rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #554444;
    display: flex;
    align-items: center;
    gap: 7px;
    box-shadow: 0 1px 2px rgba(43, 30, 30, 0.02);
}
.header-stat-pill strong {
    color: #2B1E1E;
    font-weight: 700;
}
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #2E7D4E;
    box-shadow: 0 0 0 3px rgba(46, 125, 78, 0.2);
    display: inline-block;
    animation: pulseGlow 2.4s infinite ease-in-out;
}
@keyframes pulseGlow {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.15); opacity: 0.75; }
}

/* ========================================================================= */
/* NAVIGASI SEGMENTED CONTROL (Apple / Linear Inspired)                      */
/* ========================================================================= */
.stRadio > div {
    gap: 0.5rem;
}
.stRadio [role="radiogroup"] {
    background-color: #EDE4D8 !important;
    border: 1px solid #DDD0C2 !important;
    padding: 4px !important;
    border-radius: 10px !important;
    display: inline-flex !important;
    flex-direction: row !important;
    gap: 4px !important;
    width: auto !important;
    margin-bottom: 1.5rem !important;
    box-shadow: inset 0 1px 2px rgba(43, 30, 30, 0.05) !important;
}
.stRadio [role="radiogroup"] label {
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #5C4B4B !important;
    cursor: pointer !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.15s ease !important;
}
.stRadio [role="radiogroup"] label:hover {
    background-color: rgba(255, 255, 255, 0.5) !important;
    color: #2B1E1E !important;
}
.stRadio [role="radiogroup"] label > div:first-child,
.stRadio [role="radiogroup"] input[type="radio"],
.stRadio [role="radiogroup"] [data-baseweb="radio"] input {
    display: none !important;
}
.stRadio [role="radiogroup"] label[data-checked="true"],
.stRadio [role="radiogroup"] label:has(input:checked) {
    background-color: #FFFFFF !important;
    border: 1px solid #D6C7B8 !important;
    color: #9E3B3B !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(43, 30, 30, 0.08) !important;
}

/* ========================================================================= */
/* PANEL CARDS & KONTTAINER MODERN                                           */
/* ========================================================================= */
.desktop-panel-card {
    background-color: #FFFFFF;
    border: 1px solid #E8DDD1;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 8px rgba(43, 30, 30, 0.03), 0 1px 2px rgba(43, 30, 30, 0.02);
}
.desktop-panel-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.85rem;
}
.panel-icon-wrap {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background-color: #FAF2EB;
    border: 1px solid #EEDFCE;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.desktop-panel-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #2B1E1E;
    letter-spacing: -0.01em;
    margin: 0;
}

/* ========================================================================= */
/* CUSTOM FILE UPLOADER & CAMERA STYLING                                     */
/* ========================================================================= */
[data-testid="stFileUploader"] {
    background-color: #FAF6F2 !important;
    border: 2px dashed #D9CBBE !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #9E3B3B !important;
    background-color: #FFF9F7 !important;
}
[data-testid="stFileUploader"] section {
    padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: transparent !important;
    border: none !important;
}

/* Format Berkas yang Didukung (Format Badges) */
.supported-formats-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin: 0.5rem 0 0.85rem 0;
}
.format-pill {
    background-color: #F0E6DC;
    color: #4A3C3C;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid #D9CBBE;
    letter-spacing: 0.02em;
}

/* ========================================================================= */
/* TOMBOL-TOMBOL STREAMLIT (Elevated Micro-interactions)                     */
/* ========================================================================= */
div.stButton > button {
    min-height: 46px !important;
    font-size: 0.94rem !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: 1px solid #8A2D2D !important;
    background: linear-gradient(180deg, #9E3B3B 0%, #8A2D2D 100%) !important;
    color: #FAF6F2 !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 2px 6px rgba(158, 59, 59, 0.22) !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
div.stButton > button:hover {
    background: linear-gradient(180deg, #8A2D2D 0%, #752222 100%) !important;
    border-color: #752222 !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(158, 59, 59, 0.3) !important;
}
div.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 1px 3px rgba(158, 59, 59, 0.2) !important;
}

/* Secondary Button (Pemeriksaan Foto Lain) */
div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"],
div.stButton > button[key="btn_reset_analysis"] {
    background: #FFFFFF !important;
    color: #4A3C3C !important;
    border: 1px solid #D9CBBE !important;
    box-shadow: 0 1px 3px rgba(43, 30, 30, 0.05) !important;
}
div.stButton > button[key="btn_reset_analysis"]:hover {
    background: #FAF6F2 !important;
    border-color: #9E3B3B !important;
    color: #9E3B3B !important;
}

/* ========================================================================= */
/* PANDUAN DUA PANEL CARA MEMOTRET (Visual Comparison Cards)                 */
/* ========================================================================= */
.guide-card-wrap {
    display: flex;
    flex-direction: column;
    border-radius: 10px;
    overflow: hidden;
    background-color: #FFFFFF;
    border: 1px solid #E8DDD1;
}
.guide-card-header-correct {
    background-color: #EBF5EE;
    color: #1F5A37;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 6px 10px;
    border-bottom: 1px solid #C8E3CF;
    display: flex;
    align-items: center;
    gap: 6px;
}
.guide-card-header-incorrect {
    background-color: #FDF1F0;
    color: #8C2B2B;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 6px 10px;
    border-bottom: 1px solid #F5CDCA;
    display: flex;
    align-items: center;
    gap: 6px;
}
.guide-card-body {
    padding: 8px 10px;
}
.guide-tip-item {
    font-size: 0.8rem;
    line-height: 1.4;
    color: #3E3030;
    display: flex;
    align-items: flex-start;
    gap: 6px;
    margin-bottom: 4px;
}
.guide-tip-item:last-child {
    margin-bottom: 0;
}
.guide-callout-footer {
    background-color: #FAF2EB;
    border: 1px solid #EEDFCE;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-top: 0.85rem;
    font-size: 0.82rem;
    color: #554444;
    line-height: 1.45;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ========================================================================= */
/* STREAMLIT TABS STYLING (Modern Minimalist)                               */
/* ========================================================================= */
[data-testid="stTabs"] {
    margin-top: 0.25rem;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #EEDFCE !important;
    gap: 8px !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background-color: transparent !important;
    color: #6E5C5C !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 8px 14px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    border-radius: 0 !important;
    transition: all 0.15s ease !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #2B1E1E !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #9E3B3B !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #9E3B3B !important;
}

/* Sample Gallery Card */
.sample-info-card {
    background-color: #FAF6F2;
    border: 1px solid #E8DDD1;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* ========================================================================= */
/* EMPTY STATE WORKSTATION (Clinical Welcome Space)                          */
/* ========================================================================= */
.empty-state-card {
    background-color: #FFFFFF;
    border: 1px dashed #D9CBBE;
    border-radius: 14px;
    padding: 3.5rem 2rem;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 540px;
    box-shadow: 0 2px 10px rgba(43, 30, 30, 0.02);
}
.empty-state-icon-wrap {
    width: 72px;
    height: 72px;
    border-radius: 18px;
    background: linear-gradient(135deg, #FFEAD3 0%, #FFDFBA 100%);
    border: 1px solid #F5D3B0;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1.35rem;
    box-shadow: 0 4px 14px rgba(217, 155, 38, 0.15);
}
.empty-state-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #2B1E1E;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}
.empty-state-desc {
    font-size: 0.92rem;
    color: #635252;
    max-width: 480px;
    line-height: 1.55;
    margin-bottom: 2rem;
}
.empty-features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    width: 100%;
    max-width: 640px;
}
.empty-feature-item {
    background-color: #FAF6F2;
    border: 1px solid #E8DDD1;
    border-radius: 10px;
    padding: 1rem 0.75rem;
    text-align: center;
    transition: all 0.15s ease;
}
.empty-feature-item:hover {
    background-color: #FFFFFF;
    box-shadow: 0 4px 12px rgba(43, 30, 30, 0.05);
}
.empty-feature-icon {
    margin-bottom: 6px;
    color: #9E3B3B;
}
.empty-feature-item-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #2B1E1E;
    margin-bottom: 3px;
}
.empty-feature-item-desc {
    font-size: 0.76rem;
    color: #6E5C5C;
    line-height: 1.35;
    margin: 0;
}

/* ========================================================================= */
/* HASIL DETEKSI & WORKSTATION KLINIS AKTIF                                   */
/* ========================================================================= */
.analysis-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #E8DDD1;
}
.analysis-header-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #2B1E1E;
    margin: 0;
}
.analysis-meta-pills {
    display: flex;
    gap: 6px;
}
.analysis-meta-pill {
    background-color: #FFFFFF;
    border: 1px solid #E8DDD1;
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 0.74rem;
    font-weight: 600;
    color: #6B5959;
}

/* Kartu Ringkasan Hasil (Triage Result Card) */
.result-header-card {
    background-color: #FFFFFF;
    border: 1px solid #E8DDD1;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(43, 30, 30, 0.03);
    position: relative;
    overflow: hidden;
}
.result-header-card::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 5px;
    background-color: #9E3B3B;
}
.result-header-card.is-normal::before {
    background-color: #2E7D4E;
}
.result-main-name {
    font-size: 1.55rem;
    font-weight: 800;
    color: #2B1E1E;
    letter-spacing: -0.02em;
    margin: 0 0 0.2rem 0;
}
.result-en-name {
    font-size: 0.95rem;
    color: #6E5C5C;
    font-style: italic;
    margin: 0 0 0.75rem 0;
}

/* Badges Prioritas */
.priority-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.84rem;
    font-weight: 700;
    padding: 0.3rem 0.85rem;
    border-radius: 6px;
    letter-spacing: 0.01em;
}
.priority-rendah {
    background-color: #EBF5EE;
    color: #1F5A37;
    border: 1px solid #C4E2CC;
}
.priority-sedang {
    background-color: #FFF5E6;
    color: #945900;
    border: 1px solid #F8D7A8;
}
.priority-tinggi {
    background-color: #FDF0EE;
    color: #8C2B2B;
    border: 1px solid #F5CDCA;
}
.priority-normal {
    background-color: #EBF5EE;
    color: #1F5A37;
    border: 1px solid #C4E2CC;
}

/* ========================================================================= */
/* KOMPONEN KHAS: PITA KEYAKINAN 3-ZONA (Confidence Ribbon)                  */
/* ========================================================================= */
.confidence-ribbon-container {
    margin: 1rem 0 1.35rem 0;
    padding: 1.15rem 1.25rem;
    background-color: #FFFFFF;
    border: 1px solid #E8DDD1;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(43, 30, 30, 0.03);
}
.ribbon-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}
.ribbon-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #2B1E1E;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ribbon-score-badge {
    font-size: 0.84rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
}
.ribbon-track {
    position: relative;
    height: 14px;
    border-radius: 7px;
    display: flex;
    overflow: visible;
    margin: 2.2rem 0 0.75rem 0;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.15);
}
.zone-not-detected {
    width: 29%;
    background-color: #526B7A;
    border-top-left-radius: 7px;
    border-bottom-left-radius: 7px;
    position: relative;
}
.zone-possible {
    width: 6%;
    background-color: #D99B26;
    position: relative;
}
.zone-detected {
    width: 65%;
    background-color: #9E3B3B;
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
}
.zone-detected-normal {
    width: 65%;
    background-color: #2E7D4E;
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
}
.zone-divider {
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 2px;
    background-color: rgba(255, 255, 255, 0.85);
    z-index: 2;
}
.ribbon-marker {
    position: absolute;
    top: -4px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: #2B1E1E;
    border: 3px solid #FFFFFF;
    transform: translateX(-50%);
    box-shadow: 0 2px 6px rgba(0,0,0,0.35);
    z-index: 5;
    transition: left 0.3s ease;
}
.ribbon-marker-label {
    position: absolute;
    top: -28px;
    transform: translateX(-50%);
    background-color: #2B1E1E;
    color: #FAF6F2;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 5px;
    white-space: nowrap;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}
.ribbon-marker-label::after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 50%;
    transform: translateX(-50%);
    border-width: 4px 4px 0;
    border-style: solid;
    border-color: #2B1E1E transparent;
}
.ribbon-legend {
    display: flex;
    justify-content: space-between;
    font-size: 0.76rem;
    color: #6E5C5C;
    margin-top: 0.45rem;
}
.ribbon-interpretation {
    margin-top: 0.85rem;
    padding: 0.65rem 0.9rem;
    background-color: #FAF6F2;
    border-radius: 6px;
    border: 1px solid #E8DDD1;
    font-size: 0.82rem;
    line-height: 1.45;
    color: #4A3C3C;
}

/* ========================================================================= */
/* LAYANAN DARURAT MEDIS (Emergency Hotlines Card)                           */
/* ========================================================================= */
.emergency-contacts-card {
    background-color: #FFF9F3;
    border: 1px solid #F5D3B0;
    border-radius: 10px;
    padding: 0.9rem 1.15rem;
    margin: 1rem 0;
}
.emergency-contacts-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #8C4700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.emergency-badges-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.emergency-badge {
    background-color: #FFFFFF;
    border: 1px solid #EED7C0;
    border-radius: 6px;
    padding: 0.4rem 0.75rem;
    font-size: 0.82rem;
    color: #2B1E1E;
    box-shadow: 0 1px 2px rgba(43, 30, 30, 0.03);
}
.emergency-badge strong {
    color: #9E3B3B;
    font-weight: 800;
    margin-left: 2px;
}

/* ========================================================================= */
/* LANGKAH PERTOLONGAN PERTAMA (Numbered Action Steps)                       */
/* ========================================================================= */
.first-aid-section {
    background-color: #FFFFFF;
    border: 1px solid #E8DDD1;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(43, 30, 30, 0.03);
}
.first-aid-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #F0E4D8;
}
.first-aid-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #2B1E1E;
    margin: 0;
}
.first-aid-steps-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.first-aid-step-card {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background-color: #FAF6F2;
    border: 1px solid #E8DDD1;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    transition: background-color 0.15s ease;
}
.first-aid-step-card:hover {
    background-color: #FFFBF7;
}
.step-number-badge {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, #9E3B3B 0%, #852C2C 100%);
    color: #FFFFFF;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text {
    font-size: 0.92rem;
    line-height: 1.5;
    color: #2B1E1E;
    margin: 0;
}

/* ========================================================================= */
/* KOTAK LARANGAN KRITIS (Peringatan JANGAN)                                 */
/* ========================================================================= */
.critical-warning-card {
    background-color: #FFF6F5;
    border: 1px solid #F5CDCA;
    border-left: 5px solid #9E3B3B;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}
.warning-header-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.6rem;
}
.warning-title {
    font-size: 0.94rem;
    font-weight: 800;
    color: #8C2B2B;
    margin: 0;
}
.warning-items-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.warning-item-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 0.88rem;
    line-height: 1.45;
    color: #2B1E1E;
}
.warning-bullet-icon {
    color: #9E3B3B;
    font-weight: 800;
    font-size: 0.95rem;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ========================================================================= */
/* JEJAK PENALARAN FORWARD CHAINING                                          */
/* ========================================================================= */
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E8DDD1 !important;
    border-radius: 10px !important;
    margin: 1rem 0 !important;
    box-shadow: 0 1px 3px rgba(43, 30, 30, 0.02) !important;
}
div[data-testid="stExpander"] details summary {
    font-size: 0.94rem !important;
    font-weight: 700 !important;
    color: #2B1E1E !important;
    padding: 0.75rem 1rem !important;
}
.reasoning-rule-banner {
    background-color: #FAF2EB;
    border: 1px solid #EEDFCE;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.88rem;
    color: #2B1E1E;
}
.reasoning-rule-badge {
    background-color: #9E3B3B;
    color: #FFFFFF;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
}
.reasoning-step-item {
    font-size: 0.86rem;
    line-height: 1.5;
    color: #3E3030;
    padding: 4px 0;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}

/* ========================================================================= */
/* METRIK EVALUASI MODEL (Dashboard Riset YOLOv8)                            */
/* ========================================================================= */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin: 0.75rem 0 1.5rem 0;
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
    border: 1px solid #E8DDD1;
    border-top: 3px solid #9E3B3B;
    border-radius: 10px;
    padding: 1rem 0.85rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(43, 30, 30, 0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(43, 30, 30, 0.06);
}
.metric-card-val {
    font-size: 1.55rem;
    font-weight: 800;
    color: #9E3B3B;
    letter-spacing: -0.02em;
    margin-bottom: 3px;
}
.metric-card-lbl {
    font-size: 0.82rem;
    font-weight: 700;
    color: #2B1E1E;
}
.metric-card-sub {
    font-size: 0.72rem;
    color: #6E5C5C;
    margin-top: 3px;
}

/* Tabel Metrik Performa */
.metric-table-container {
    overflow-x: auto;
    margin: 0.75rem 0 1.35rem 0;
    border: 1px solid #E8DDD1;
    border-radius: 10px;
    background-color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(43, 30, 30, 0.03);
}
.metric-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
}
.metric-table th {
    background-color: #FAF2EB;
    color: #2B1E1E;
    font-weight: 700;
    padding: 10px 14px;
    border-bottom: 1px solid #E8DDD1;
    text-align: left;
    white-space: nowrap;
}
.metric-table td {
    padding: 9px 14px;
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
.metric-table tr:hover td {
    background-color: #F8F1EB;
}

/* Dataset Distribution Progress Bar */
.dataset-bar-wrap {
    height: 12px;
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    margin: 0.85rem 0 0.65rem 0;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
}
.dataset-bar-train {
    background-color: #9E3B3B;
    width: 70%;
}
.dataset-bar-valid {
    background-color: #D99B26;
    width: 20%;
}
.dataset-bar-test {
    background-color: #526B7A;
    width: 10%;
}

/* ========================================================================= */
/* HALAMAN TENTANG & FOOTER                                                  */
/* ========================================================================= */
.about-card {
    background-color: #FFFFFF;
    border: 1px solid #E8DDD1;
    border-radius: 12px;
    padding: 1.35rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 8px rgba(43, 30, 30, 0.03);
}
.about-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #9E3B3B;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.about-card-body {
    font-size: 0.9rem;
    line-height: 1.6;
    color: #3E3030;
    margin: 0;
}
.footer-disclaimer-box {
    margin-top: 2.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid #E8DDD1;
    font-size: 0.82rem;
    line-height: 1.55;
    color: #6E5C5C;
    text-align: center;
}
.footer-disclaimer-box a {
    color: #9E3B3B;
    text-decoration: underline;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Komponen Visual Khas: Pita Keyakinan (Three-zone Horizontal Bar)
# -----------------------------------------------------------------------------
def render_confidence_ribbon(confidence: float, status: str, is_normal: bool = False):
    percent = round(confidence * 100, 1)
    clamped_pos = max(2.5, min(97.5, percent))
    zone3_class = "zone-detected-normal" if is_normal else "zone-detected"

    if percent >= 35.0:
        badge_bg = "#EBF5EE" if is_normal else "#FDF0EE"
        badge_color = "#1F5A37" if is_normal else "#8C2B2B"
        badge_text = "Zona 3: Terdeteksi (≥ 35%)"
        explanation = f"✓ Nilai keyakinan model <strong>{percent}%</strong> berada pada zona terdeteksi klinis (≥ 35%). Sistem mengonfirmasi pola visual dan memicu penalaran Forward Chaining."
    elif percent >= 29.0:
        badge_bg = "#FFF5E6"
        badge_color = "#945900"
        badge_text = "Zona 2: Kemungkinan (29%–35%)"
        explanation = f"⚠️ Nilai keyakinan model <strong>{percent}%</strong> berada pada ambang batas antara (29%–35%). Lesi dicurigai ada namun disarankan melakukan pemotretan ulang dengan pencahayaan lebih jelas."
    else:
        badge_bg = "#EEF3F6"
        badge_color = "#395261"
        badge_text = "Zona 1: Tidak Terdeteksi (< 29%)"
        explanation = f"ℹ️ Nilai keyakinan model <strong>{percent}%</strong> berada di bawah batas sensitivitas (&lt; 29%). Citra mungkin terlalu buram, jauh, atau permukaan kulit tidak memperlihatkan kelainan patologis."

    html = f"""
    <div class="confidence-ribbon-container">
        <div class="ribbon-header-row">
            <div class="ribbon-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                Pita Tingkat Keyakinan Model
            </div>
            <div class="ribbon-score-badge tabular-number" style="background-color: {badge_bg}; color: {badge_color};">
                {badge_text}
            </div>
        </div>
        <div class="ribbon-track">
            <div class="zone-not-detected" title="Tidak terdeteksi (&lt; 29%)">
                <div class="zone-divider"></div>
            </div>
            <div class="zone-possible" title="Kemungkinan (29% - 35%)">
                <div class="zone-divider"></div>
            </div>
            <div class="{zone3_class}" title="Terdeteksi (&ge; 35%)"></div>
            <div class="ribbon-marker" style="left: {clamped_pos}%;">
                <div class="ribbon-marker-label tabular-number">{percent}%</div>
            </div>
        </div>
        <div class="ribbon-legend">
            <span>Zona 1: &lt; 29% (Rendah)</span>
            <span>Zona 2: 29%–35% (Ambang)</span>
            <span>Zona 3: &ge; 35% (Tinggi)</span>
        </div>
        <div class="ribbon-interpretation">
            {explanation}
        </div>
    </div>
    """
    safe_markdown(html)

# -----------------------------------------------------------------------------
# Header Aplikasi dengan Lambang Medis Resmi & Status Aktif
# -----------------------------------------------------------------------------
header_html = f"""
<div class="app-header-card">
    <div class="app-header-inner">
        <div class="app-header-left">
            <div class="app-emblem" aria-hidden="true">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="10.25" y="3.5" width="3.5" height="17" rx="1.75" fill="#FAF6F2"/>
                    <rect x="3.5" y="10.25" width="17" height="3.5" rx="1.75" fill="#FAF6F2"/>
                </svg>
            </div>
            <div class="app-title-wrap">
                <div class="app-title-row">
                    <h1 class="app-title">{APP_CONFIG.get('app', {}).get('title', 'Deteksi Luka Kulit & Rekomendasi P3K')}</h1>
                    <span class="app-badge-version">Sistem Pakar &amp; AI</span>
                </div>
                <p class="app-subtitle">{APP_CONFIG.get('app', {}).get('subtitle', 'Sistem panduan pertolongan pertama awal berbasis deteksi visual YOLOv8 dan penalaran Forward Chaining')}</p>
            </div>
        </div>
        <div class="header-stat-pills">
            <div class="header-stat-pill">
                <span class="pulse-dot"></span>
                <span>YOLOv8 ONNX: <strong>Aktif</strong></span>
            </div>
            <div class="header-stat-pill">
                <span>mAP@0.5: <strong>86.4%</strong></span>
            </div>
            <div class="header-stat-pill">
                <span>Aturan Pakar: <strong>R001–R005</strong></span>
            </div>
        </div>
    </div>
</div>
"""
safe_markdown(header_html)

# Navigasi Halaman Utama (Segmented Tab Bar)
current_page = st.radio(
    label="Pilih menu navigasi",
    options=["Pemeriksaan luka", "Metrik evaluasi model", "Tentang sistem"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

# -----------------------------------------------------------------------------
# Halaman 1: Pemeriksaan Luka (Alur Kerja Triase & Rekomendasi P3K)
# -----------------------------------------------------------------------------
if current_page == "Pemeriksaan luka":
    if "chosen_example_path" not in st.session_state:
        st.session_state["chosen_example_path"] = None

    col_workspace, col_analysis = st.columns([1, 1.25], gap="large")

    with col_workspace:
        # 1. Panel Masukan Foto Kulit
        safe_markdown("""
        <div class="desktop-panel-card">
            <div class="desktop-panel-header">
                <div class="panel-icon-wrap">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                </div>
                <div>
                    <div class="desktop-panel-title">Masukan Foto Kulit</div>
                    <div style="font-size: 0.78rem; color: #6E5C5C;">Pilih berkas dari perangkat atau ambil foto langsung</div>
                </div>
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
            camera_file = st.camera_input("Arahkan kamera ke area luka kulit (Jarak 10-20 cm)")
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
                <span style="font-size: 0.78rem; font-weight: 700; color: #554444;">Format:</span>
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

        # 2. Panel Panduan Cara Memotret (Visual Comparison)
        safe_markdown("""
        <div class="desktop-panel-card" style="margin-top: 1.15rem;">
            <div class="desktop-panel-header">
                <div class="panel-icon-wrap">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                        <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                </div>
                <div>
                    <div class="desktop-panel-title">Panduan Pengambilan Foto</div>
                    <div style="font-size: 0.78rem; color: #6E5C5C;">Patuhi panduan jarak dan fokus untuk hasil deteksi optimal</div>
                </div>
            </div>
        </div>
        """)

        col_cam_benar, col_cam_salah = st.columns(2, gap="medium")
        img_b_path = resolve_sample_path("app/static/cara_memotret_benar.svg")
        if not os.path.exists(img_b_path):
            img_b_path = resolve_sample_path("static/cara_memotret_benar.svg")

        img_s_path = resolve_sample_path("app/static/cara_memotret_salah.svg")
        if not os.path.exists(img_s_path):
            img_s_path = resolve_sample_path("static/cara_memotret_salah.svg")

        with col_cam_benar:
            safe_markdown("""
            <div class="guide-card-wrap">
                <div class="guide-card-header-correct">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    <span>SEPERTI INI (Dianjurkan)</span>
                </div>
            </div>
            """)
            st.image(img_b_path, use_container_width=True)
            safe_markdown("""
            <div class="guide-card-body" style="background:#FFFFFF; border:1px solid #E8DDD1; border-top:none; border-radius:0 0 10px 10px;">
                <div class="guide-tip-item"><span style="color:#1F5A37; font-weight:700;">✓</span> Jarak 10–20 cm dari permukaan kulit</div>
                <div class="guide-tip-item"><span style="color:#1F5A37; font-weight:700;">✓</span> Satu luka memenuhi bingkai foto</div>
                <div class="guide-tip-item"><span style="color:#1F5A37; font-weight:700;">✓</span> Fokus tajam & pencahayaan terang</div>
            </div>
            """)

        with col_cam_salah:
            safe_markdown("""
            <div class="guide-card-wrap">
                <div class="guide-card-header-incorrect">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    <span>HINDARI (Tidak Dianjurkan)</span>
                </div>
            </div>
            """)
            st.image(img_s_path, use_container_width=True)
            safe_markdown("""
            <div class="guide-card-body" style="background:#FFFFFF; border:1px solid #E8DDD1; border-top:none; border-radius:0 0 10px 10px;">
                <div class="guide-tip-item"><span style="color:#8C2B2B; font-weight:700;">✕</span> Terlalu jauh atau luka berukuran kecil</div>
                <div class="guide-tip-item"><span style="color:#8C2B2B; font-weight:700;">✕</span> Foto buram, goyang, atau tidak fokus</div>
                <div class="guide-tip-item"><span style="color:#8C2B2B; font-weight:700;">✕</span> Terhalang bayangan gelap atau silau</div>
            </div>
            """)

        safe_markdown("""
        <div class="guide-callout-footer">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#D99B26" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <span><strong>Tips Klinis:</strong> Pengambilan gambar yang stabil dan tajam meningkatkan keakuratan estimasi model hingga &gt; 85%.</span>
        </div>
        """)

        # 3. Panel Uji Cepat Sampel Citra Penelitian
        safe_markdown("""
        <div class="desktop-panel-card" style="margin-top: 1.15rem;">
            <div class="desktop-panel-header">
                <div class="panel-icon-wrap">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                        <polyline points="2 17 12 22 22 17"></polyline>
                        <polyline points="2 12 12 17 22 12"></polyline>
                    </svg>
                </div>
                <div>
                    <div class="desktop-panel-title">Uji Cepat Sampel Citra Riil</div>
                    <div style="font-size: 0.78rem; color: #6E5C5C;">Pilih sampel dari 1.391 dataset penelitian untuk menguji sistem</div>
                </div>
            </div>
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
            col_img, col_desc = st.columns([1, 1.25], gap="medium")
            with col_img:
                st.image(resolve_sample_path("sample_images/Abrasions_5.jpg"), caption="Sampel Luka Lecet (Abrasion)", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div class="sample-info-card">
                    <div>
                        <div style="font-size: 1rem; font-weight: 700; color: #2B1E1E;">Luka lecet (Abrasion)</div>
                        <div style="margin: 4px 0 8px 0;"><span class="priority-badge priority-rendah">Prioritas Rendah</span></div>
                        <p style="font-size: 0.82rem; line-height: 1.45; color: #4A3C3C; margin: 0 0 8px 0;">
                            Epidermis tergores atau terkelupas akibat gesekan mekanik tanpa perdarahan deras.
                        </p>
                    </div>
                </div>
                """)
                if st.button("Uji deteksi foto ini", key="btn_sample_abr"):
                    st.session_state["chosen_example_path"] = resolve_sample_path("sample_images/Abrasions_5.jpg")
                    st.rerun()

        with tab_bru:
            col_img, col_desc = st.columns([1, 1.25], gap="medium")
            with col_img:
                st.image(resolve_sample_path("sample_images/Bruise_11.jpg"), caption="Sampel Luka Memar (Bruise)", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div class="sample-info-card">
                    <div>
                        <div style="font-size: 1rem; font-weight: 700; color: #2B1E1E;">Luka memar (Bruise)</div>
                        <div style="margin: 4px 0 8px 0;"><span class="priority-badge priority-rendah">Prioritas Rendah</span></div>
                        <p style="font-size: 0.82rem; line-height: 1.45; color: #4A3C3C; margin: 0 0 8px 0;">
                            Perubahan warna jaringan subkutan (kebiruan/keunguan) akibat pecahnya kapiler tanpa luka terbuka.
                        </p>
                    </div>
                </div>
                """)
                if st.button("Uji deteksi foto ini", key="btn_sample_bru"):
                    st.session_state["chosen_example_path"] = resolve_sample_path("sample_images/Bruise_11.jpg")
                    st.rerun()

        with tab_bur:
            col_img, col_desc = st.columns([1, 1.25], gap="medium")
            with col_img:
                st.image(resolve_sample_path("sample_images/Burn_1.jpg"), caption="Sampel Luka Bakar (Burn)", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div class="sample-info-card">
                    <div>
                        <div style="font-size: 1rem; font-weight: 700; color: #2B1E1E;">Luka bakar (Burn)</div>
                        <div style="margin: 4px 0 8px 0;"><span class="priority-badge priority-sedang">Prioritas Sedang / Tinggi</span></div>
                        <p style="font-size: 0.82rem; line-height: 1.45; color: #4A3C3C; margin: 0 0 8px 0;">
                            Eritema meluas, jaringan melepuh atau timbul bula berisi cairan akibat panas atau api.
                        </p>
                    </div>
                </div>
                """)
                if st.button("Uji deteksi foto ini", key="btn_sample_bur"):
                    st.session_state["chosen_example_path"] = resolve_sample_path("sample_images/Burn_1.jpg")
                    st.rerun()

        with tab_cut:
            col_img, col_desc = st.columns([1, 1.25], gap="medium")
            with col_img:
                st.image(resolve_sample_path("sample_images/Cut_4.jpg"), caption="Sampel Luka Sayat (Cut)", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div class="sample-info-card">
                    <div>
                        <div style="font-size: 1rem; font-weight: 700; color: #2B1E1E;">Luka sayat (Cut)</div>
                        <div style="margin: 4px 0 8px 0;"><span class="priority-badge priority-sedang">Prioritas Sedang / Tinggi</span></div>
                        <p style="font-size: 0.82rem; line-height: 1.45; color: #4A3C3C; margin: 0 0 8px 0;">
                            Garis sayatan atau robekan linier bertepi tegas akibat benda tajam dengan resiko perdarahan aktif.
                        </p>
                    </div>
                </div>
                """)
                if st.button("Uji deteksi foto ini", key="btn_sample_cut"):
                    st.session_state["chosen_example_path"] = resolve_sample_path("sample_images/Cut_4.jpg")
                    st.rerun()

        with tab_nor:
            col_img, col_desc = st.columns([1, 1.25], gap="medium")
            with col_img:
                st.image(resolve_sample_path("sample_images/no_abnormality_2.jpg"), caption="Sampel Kulit Normal", use_container_width=True)
            with col_desc:
                safe_markdown("""
                <div class="sample-info-card">
                    <div>
                        <div style="font-size: 1rem; font-weight: 700; color: #2B1E1E;">Kulit normal (No Abnormality)</div>
                        <div style="margin: 4px 0 8px 0;"><span class="priority-badge priority-normal">Prioritas Normal</span></div>
                        <p style="font-size: 0.82rem; line-height: 1.45; color: #4A3C3C; margin: 0 0 8px 0;">
                            Sawar kulit utuh dan sehat alami tanpa lesi patologis, pembengkakan, maupun perdarahan.
                        </p>
                    </div>
                </div>
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

    # -------------------------------------------------------------------------
    # Kolom Kanan: Workstation Analisis Klinis & Rekomendasi P3K
    # -------------------------------------------------------------------------
    with col_analysis:
        if raw_image is None:
            safe_markdown("""
            <div class="empty-state-card">
                <div class="empty-state-icon-wrap">
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                    </svg>
                </div>
                <div class="empty-state-title">Ruang Analisis &amp; Rekomendasi Medis</div>
                <div class="empty-state-desc">
                    Silakan unggah foto luka kulit atau pilih salah satu sampel citra dari panel sebelah kiri. Sistem akan langsung mendeteksi lesi dan menyimpulkan pertolongan pertama berbasis aturan medis.
                </div>
                <div class="empty-features-grid">
                    <div class="empty-feature-item">
                        <div class="empty-feature-icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                        </div>
                        <div class="empty-feature-item-title">Deteksi YOLOv8</div>
                        <div class="empty-feature-item-desc">5 Kategori lesi kulit dengan presisi tinggi</div>
                    </div>
                    <div class="empty-feature-item">
                        <div class="empty-feature-icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
                        </div>
                        <div class="empty-feature-item-title">Pita Keyakinan</div>
                        <div class="empty-feature-item-desc">3 Zona ambang batas (&lt;29%, 29-35%, &ge;35%)</div>
                    </div>
                    <div class="empty-feature-item">
                        <div class="empty-feature-icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                        </div>
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

            safe_markdown("""
            <div class="analysis-header-bar">
                <div class="analysis-header-title">Hasil Analisis &amp; Triase Klinis</div>
                <div class="analysis-meta-pills">
                    <span class="analysis-meta-pill">YOLOv8 Nano</span>
                    <span class="analysis-meta-pill">640×640 px</span>
                    <span class="analysis-meta-pill">Inferensi ~15 ms</span>
                </div>
            </div>
            """)

            st.image(
                annotated_rgb,
                caption="Citra anotasi bounding box lokalisasi lesi YOLOv8",
                use_container_width=True
            )

            label_id = triage.get("label_id", top_detection.get("label_id", "Kulit normal"))
            label_en = triage.get("label_en", top_detection.get("label_en", "No Abnormality"))
            priority = triage.get("priority", "Rendah")
            confidence = float(top_detection.get("confidence", 0.0))
            status = top_detection.get("status", "detected")
            is_normal = "normal" in label_en.lower() or "abnormality" in label_en.lower()

            priority_class = "priority-rendah"
            priority_icon = "🟢"
            if priority.lower() == "sedang":
                priority_class = "priority-sedang"
                priority_icon = "🟡"
            elif priority.lower() in ["tinggi", "gawat darurat"]:
                priority_class = "priority-tinggi"
                priority_icon = "🔴"
            elif is_normal or priority.lower() == "normal":
                priority_class = "priority-normal"
                priority_icon = "⚪"

            card_normal_modifier = " is-normal" if is_normal else ""

            result_header_html = f"""
            <div class="result-header-card{card_normal_modifier}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap;">
                    <div>
                        <h2 class="result-main-name">{label_id}</h2>
                        <div class="result-en-name">{label_en}</div>
                    </div>
                    <div>
                        <span class="priority-badge {priority_class}">
                            <span>{priority_icon}</span>
                            <span>Tingkat Prioritas: {priority}</span>
                        </span>
                    </div>
                </div>
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
                        <div class="emergency-contacts-title">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8C4700" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                            </svg>
                            <span>Layanan Darurat Medis Cepat (Indonesia)</span>
                        </div>
                        <div class="emergency-badges-wrap">
                            <span class="emergency-badge">Ambulans: <strong>119</strong></span>
                            <span class="emergency-badge">Panggilan Darurat: <strong>112</strong></span>
                            <span class="emergency-badge">PMI: <strong>021-7992325</strong></span>
                        </div>
                    </div>
                    """
                )

            # Langkah Pertolongan Pertama (First Aid Steps Cards)
            actions = triage.get("actions", [])
            if actions:
                step_cards_html = ""
                for idx, act in enumerate(actions, start=1):
                    step_cards_html += f"""
                    <div class="first-aid-step-card">
                        <div class="step-number-badge">{idx}</div>
                        <div class="step-text">{act}</div>
                    </div>
                    """

                first_aid_html = f"""
                <div class="first-aid-section">
                    <div class="first-aid-header">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <div class="first-aid-title">Langkah Pertolongan Pertama (P3K)</div>
                    </div>
                    <div class="first-aid-steps-grid">
                        {step_cards_html}
                    </div>
                </div>
                """
                safe_markdown(first_aid_html)

            # Penekanan Visual Larangan Kritis (Peringatan JANGAN)
            critical_warnings = triage.get("critical_warnings", [])
            if critical_warnings:
                warning_items_html = ""
                for w in critical_warnings:
                    warning_items_html += f"""
                    <div class="warning-item-row">
                        <span class="warning-bullet-icon">✕</span>
                        <span>{w}</span>
                    </div>
                    """

                warning_html = f"""
                <div class="critical-warning-card">
                    <div class="warning-header-row">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                            <line x1="12" y1="9" x2="12" y2="13"></line>
                            <line x1="12" y1="17" x2="12.01" y2="17"></line>
                        </svg>
                        <div class="warning-title">Perhatian Kritis (Larangan Penanganan)</div>
                    </div>
                    <div class="warning-items-list">
                        {warning_items_html}
                    </div>
                </div>
                """
                safe_markdown(warning_html)

            # Jejak Penalaran Forward Chaining (Pipeline Log)
            with st.expander("Jejak penalaran sistem pakar (Forward Chaining)"):
                rule_id = triage.get("rule_id", "R005")
                safe_markdown(f"""
                <div class="reasoning-rule-banner">
                    <span class="reasoning-rule-badge">{rule_id}</span>
                    <span><strong>Aturan Aktif:</strong> Aturan Pertolongan Pertama untuk {label_id} ({label_en})</span>
                </div>
                """)
                for step in triage.get("reasoning_steps", []):
                    safe_markdown(f"""
                    <div class="reasoning-step-item">
                        <span style="color: #9E3B3B; font-weight: 700;">•</span>
                        <span>{step}</span>
                    </div>
                    """)
                st.markdown(f"**Kesimpulan Inferensi:** {triage.get('conclusion', '')}")
                if triage.get("notes"):
                    st.info(f"Catatan Medis: {triage.get('notes')}")

            # Tombol Reset / Analisis Baru
            if st.button("Pemeriksaan foto lain", key="btn_reset_analysis"):
                st.session_state["chosen_example_path"] = None
                st.rerun()

    # Catatan Kaki & Penafian Singkat yang Tenang (Spanning full width)
    safe_markdown(
        """
        <div class="footer-disclaimer-box">
            Sistem ini mendeteksi 5 kategori permukaan kulit (luka lecet, luka memar, luka bakar, luka sayat, dan kulit normal).
            Hasil pemeriksaan berupa panduan pertolongan pertama awal dan bukan pengganti diagnosis tenaga medis profesional.
        </div>
        """
    )

# -----------------------------------------------------------------------------
# Halaman 2: Metrik Evaluasi Model (Kinerja Riset YOLOv8)
# -----------------------------------------------------------------------------
elif current_page == "Metrik evaluasi model":
    safe_markdown("""
    <div style="margin: 0.5rem 0 1.25rem 0;">
        <h2 style="font-size: 1.45rem; font-weight: 800; color: #2B1E1E; letter-spacing: -0.02em; margin: 0 0 4px 0;">Metrik Evaluasi Model Deteksi</h2>
        <p style="font-size: 0.88rem; color: #6E5C5C; margin: 0;">Kinerja kuantitatif model YOLOv8 pada External Wound Dataset (1.391 citra) sesuai Subbab 3.5 Proposal Tugas Akhir</p>
    </div>
    """)

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
                <div class="metric-card-sub">Rerata Multi-Ambang</div>
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
                <div class="metric-card-lbl">Waktu Inferensi</div>
                <div class="metric-card-sub">Setara ~{metrics_ov.get('fps', 67.5)} FPS</div>
            </div>
        </div>
        """
    )

    # 2. Layout Desktop Dua Kolom untuk Evaluasi Mendalam
    col_p2_left, col_p2_right = st.columns([1.15, 0.85], gap="large")

    with col_p2_left:
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; color: #2B1E1E; margin: 0 0 0.5rem 0;'>Kinerja Deteksi per Kategori Lesi Kulit</h3>", unsafe_allow_html=True)
        table_rows = ""
        for raw_c, data in metrics_cls.items():
            lbl_id = data.get("label_id", raw_c)
            table_rows += f"""
            <tr>
                <td><strong>{lbl_id}</strong> <span style="color: #6E5C5C;">({raw_c})</span></td>
                <td class="tabular-number">{data.get('precision', 0) * 100:.1f}%</td>
                <td class="tabular-number">{data.get('recall', 0) * 100:.1f}%</td>
                <td class="tabular-number">{data.get('f1_score', 0) * 100:.1f}%</td>
                <td class="tabular-number" style="font-weight: 800; color: #9E3B3B;">{data.get('mAP_50', 0) * 100:.1f}%</td>
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
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; color: #2B1E1E; margin: 0 0 0.5rem 0;'>Matriks Konfusi Data Uji</h3>", unsafe_allow_html=True)
        cm_labels = conf_mat.get("labels", [])
        cm_matrix = conf_mat.get("matrix", [])

        cm_th = "".join([f"<th>{l}</th>" for l in cm_labels])
        cm_rows = ""
        for i, row in enumerate(cm_matrix):
            actual_label = cm_labels[i] if i < len(cm_labels) else f"Kelas {i}"
            cells = ""
            for j, val in enumerate(row):
                is_diagonal = (i == j)
                bg_style = ' style="background-color: #EBF5EE; font-weight: 800; color: #1F5A37;"' if is_diagonal else ''
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

        splits = dataset_info.get("splits", {})
        safe_markdown(
            f"""
            <div class="about-card" style="margin-top: 0;">
                <div class="about-card-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                    <span>Distribusi Dataset Penelitian (1.391 Citra)</span>
                </div>
                <p class="about-card-body" style="font-size: 0.85rem;">
                    Model dilatih menggunakan dataset gabungan <strong>External Wound Dataset</strong> (Roboflow Universe &amp; Kaggle) yang dibagi secara proporsional:
                </p>
                <div class="dataset-bar-wrap">
                    <div class="dataset-bar-train" title="Train: 70%"></div>
                    <div class="dataset-bar-valid" title="Valid: 20%"></div>
                    <div class="dataset-bar-test" title="Test: 10%"></div>
                </div>
                <div style="display: flex; gap: 8px; margin-top: 0.75rem; flex-wrap: wrap;">
                    <div style="background-color: #FAF6F2; border: 1px solid #E8DDD1; border-radius: 6px; padding: 0.55rem 0.75rem; flex: 1; min-width: 95px;">
                        <div style="font-size: 0.74rem; color: #6E5C5C;">Latih (Train)</div>
                        <div class="tabular-number" style="font-size: 1.1rem; font-weight: 800; color: #9E3B3B;">{splits.get('train', 979)} <span style="font-size: 0.78rem; font-weight: 500; color: #6E5C5C;">(70%)</span></div>
                    </div>
                    <div style="background-color: #FAF6F2; border: 1px solid #E8DDD1; border-radius: 6px; padding: 0.55rem 0.75rem; flex: 1; min-width: 95px;">
                        <div style="font-size: 0.74rem; color: #6E5C5C;">Validasi (Valid)</div>
                        <div class="tabular-number" style="font-size: 1.1rem; font-weight: 800; color: #D99B26;">{splits.get('valid', 275)} <span style="font-size: 0.78rem; font-weight: 500; color: #6E5C5C;">(20%)</span></div>
                    </div>
                    <div style="background-color: #FAF6F2; border: 1px solid #E8DDD1; border-radius: 6px; padding: 0.55rem 0.75rem; flex: 1; min-width: 95px;">
                        <div style="font-size: 0.74rem; color: #6E5C5C;">Uji (Test)</div>
                        <div class="tabular-number" style="font-size: 1.1rem; font-weight: 800; color: #526B7A;">{splits.get('test', 137)} <span style="font-size: 0.78rem; font-weight: 500; color: #6E5C5C;">(10%)</span></div>
                    </div>
                </div>
            </div>
            """
        )

        with st.expander("Cara mengecek metrik mandiri via terminal (CLI)"):
            st.markdown(
                textwrap.dedent(
                    """
                    Jalankan perintah berikut di PowerShell atau terminal:
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
    safe_markdown("""
    <div style="margin: 0.5rem 0 1.25rem 0;">
        <h2 style="font-size: 1.45rem; font-weight: 800; color: #2B1E1E; letter-spacing: -0.02em; margin: 0 0 4px 0;">Tentang Sistem Deteksi &amp; Rekomendasi P3K</h2>
        <p style="font-size: 0.88rem; color: #6E5C5C; margin: 0;">Informasi arsitektur, atribusi dataset, komitmen privasi, serta batas kemampuan sistem</p>
    </div>
    """)

    col_about_1, col_about_2 = st.columns(2, gap="large")

    with col_about_1:
        safe_markdown(
            f"""
            <div class="about-card">
                <div class="about-card-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                        <polyline points="2 17 12 22 22 17"></polyline>
                        <polyline points="2 12 12 17 22 12"></polyline>
                    </svg>
                    <span>Arsitektur &amp; Cara Kerja Sistem</span>
                </div>
                <p class="about-card-body">
                    {APP_CONFIG.get('about', {}).get('system_description',
                    'Sistem ini mengintegrasikan model deteksi visual berbasis YOLOv8 (ONNX) untuk melokalisasi area luka dan menghitung skor keyakinan, dengan mesin penalaran terstruktur Forward Chaining (aturan R001-R005) guna menyimpulkan tingkat keparahan serta urutan langkah pertolongan pertama.')}
                </p>
            </div>
            """
        )

        safe_markdown(
            f"""
            <div class="about-card">
                <div class="about-card-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                    </svg>
                    <span>Sumber Data &amp; Atribusi Penelitian</span>
                </div>
                <p class="about-card-body">
                    {APP_CONFIG.get('about', {}).get('dataset_attribution',
                    'Dataset penelitian bersumber dari External Wound Dataset (Roboflow Universe & Kaggle) berjumlah 1.391 citra lesi kulit yang dianotasi ke dalam 5 kelas.')}
                </p>
            </div>
            """
        )

    with col_about_2:
        safe_markdown(
            f"""
            <div class="about-card">
                <div class="about-card-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                    </svg>
                    <span>Pernyataan Privasi Pengguna</span>
                </div>
                <p class="about-card-body">
                    {APP_CONFIG.get('about', {}).get('privacy_statement',
                    'Foto yang Anda ambil atau unggah diproses secara langsung di memori lokal saat sesi berjalan. Sistem tidak menyimpan citra di penyimpanan permanen atau basis data eksternal, dan foto langsung dibersihkan setelah proses analisis.')}
                </p>
            </div>
            """
        )

        safe_markdown(
            f"""
            <div class="about-card" style="border-left: 5px solid #9E3B3B;">
                <div class="about-card-title" style="color: #9E3B3B;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9E3B3B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                        <line x1="12" y1="9" x2="12" y2="13"></line>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                    <span>Peringatan &amp; Penafian Medis Resmi</span>
                </div>
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
            Aplikasi ini dikembangkan untuk keperluan penelitian akademis Tugas Akhir dan edukasi pertolongan pertama (P3K).
        </div>
        """
    )
