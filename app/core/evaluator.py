"""
Modul Evaluasi Model & Metrik Penelitian (Subbab 3.5 Proposal)

Sumber kebenaran TUNGGAL untuk metrik: berkas reports/metrics.json yang dihasilkan oleh
`python scripts/evaluate_full.py`. Tidak ada angka yang diketik manual di sini.

Modul ini dipakai oleh:
  - antarmuka Streamlit (tab Evaluasi Model),
  - scripts/check_metrics.py,
  - Flask API lama (core/evaluator.py me-re-export modul ini).
"""

import json
import os
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

CLASS_LABELS = ["Abrasions", "Bruise", "Burn", "Cut", "No Abnormality"]


def load_benchmark(path: str = METRICS_PATH) -> Optional[Dict[str, Any]]:
    """Membaca metrics.json. Mengembalikan None bila belum pernah dijalankan."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Alias kompatibilitas mundur (API lama merujuk ke data uji)
    ev = data.get("evaluations", {})
    ref = ev.get("testing") or (list(ev.values())[-1] if ev else None)
    if ref:
        data["metrics_overall"] = ref["metrics_overall"]
        data["metrics_per_class"] = ref["metrics_per_class"]
        data["confusion_matrix"] = ref["confusion_matrix"]
    return data


BENCHMARK_RESULTS: Dict[str, Any] = load_benchmark() or {}


class EvaluationEngine:
    @staticmethod
    def is_available() -> bool:
        return bool(BENCHMARK_RESULTS)

    @staticmethod
    def get_benchmark_report(reload: bool = False) -> Dict[str, Any]:
        """Mengambil data laporan evaluasi lengkap untuk visualisasi dashboard."""
        global BENCHMARK_RESULTS
        if reload or not BENCHMARK_RESULTS:
            BENCHMARK_RESULTS = load_benchmark() or {}
        return BENCHMARK_RESULTS

    @staticmethod
    def figure_path(name: str) -> Optional[str]:
        """Path absolut gambar hasil evaluasi (mis. 'cm_test', 'pr', 'sweep'), atau None."""
        figs = EvaluationEngine.get_benchmark_report().get("figures", {})
        rel = figs.get(name)
        if not rel:
            return None
        p = os.path.join(BASE_DIR, rel)
        return p if os.path.exists(p) else None

    @staticmethod
    def report_markdown_path() -> Optional[str]:
        p = os.path.join(REPORT_DIR, "LAPORAN_EVALUASI.md")
        return p if os.path.exists(p) else None
