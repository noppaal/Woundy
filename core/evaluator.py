"""
Evaluator untuk server Flask lama (app.py di root).

Seluruh metrik dibaca dari reports/metrics.json (sumber tunggal, lihat app/core/evaluator.py).
Fungsi bantu IoU dan Precision/Recall/F1 dipertahankan untuk kompatibilitas.
Ahmad Naufal Ramadhan - Universitas Telkom 2026
"""

import importlib.util
import os
from typing import Any, Dict, List

_APP_EVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "core", "evaluator.py")
_spec = importlib.util.spec_from_file_location("_app_core_evaluator", _APP_EVAL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CLASS_LABELS = _mod.CLASS_LABELS
BENCHMARK_RESULTS: Dict[str, Any] = _mod.BENCHMARK_RESULTS
load_benchmark = _mod.load_benchmark


class EvaluationEngine(_mod.EvaluationEngine):
    """Mesin kalkulasi metrik evaluasi deteksi objek standar (IoU, P, R, F1)."""

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Intersection over Union (Persamaan 3.1): IoU = Area(A ∩ B) / Area(A ∪ B)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
        """
        Precision (Persamaan 3.5): TP / (TP + FP)
        Recall (Persamaan 3.6): TP / (TP + FN)
        F1-Score (Persamaan 3.7): 2 * (P * R) / (P + R)
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1_score": round(f1, 4)}
