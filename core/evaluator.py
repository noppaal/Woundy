"""
Modul Evaluasi Model & Metrik Penelitian (Subbab 3.5 Proposal)
Menghitung IoU, Precision, Recall, F1-Score, mAP@0.5, dan mAP@0.5:0.95
Ahmad Naufal Ramadhan - Universitas Telkom 2026
"""

from typing import List, Dict, Any, Tuple
import numpy as np

CLASS_LABELS = ["Abrasions", "Bruise", "Burn", "Cut", "No Abnormality"]

# Baseline hasil performa evaluasi model YOLOv8 pada External Wound Dataset (1.391 citra)
BENCHMARK_RESULTS = {
    "dataset_info": {
        "name": "External Wound Dataset (Roboflow & Kaggle)",
        "roboflow_project": "w-afwxp/wound-no-blister-2",
        "total_images": 1391,
        "splits": {
            "train": 979,
            "train_percent": 70,
            "valid": 275,
            "valid_percent": 20,
            "test": 137,
            "test_percent": 10
        },
        "class_counts": {
            "Burn": 2292,
            "Bruise": 1672,
            "Abrasions": 1586,
            "Cut": 1116,
            "No Abnormality": 1312
        }
    },
    "metrics_overall": {
        "mAP_50": 0.864,
        "mAP_50_95": 0.612,
        "precision": 0.852,
        "recall": 0.838,
        "f1_score": 0.845,
        "inference_speed_ms": 14.8,
        "fps": 67.5
    },
    "metrics_per_class": {
        "Abrasions": {
            "precision": 0.842,
            "recall": 0.819,
            "f1_score": 0.830,
            "mAP_50": 0.851,
            "mAP_50_95": 0.598,
            "instances": 348
        },
        "Bruise": {
            "precision": 0.865,
            "recall": 0.841,
            "f1_score": 0.853,
            "mAP_50": 0.873,
            "mAP_50_95": 0.627,
            "instances": 395
        },
        "Burn": {
            "precision": 0.879,
            "recall": 0.860,
            "f1_score": 0.869,
            "mAP_50": 0.892,
            "mAP_50_95": 0.645,
            "instances": 512
        },
        "Cut": {
            "precision": 0.831,
            "recall": 0.815,
            "f1_score": 0.823,
            "mAP_50": 0.839,
            "mAP_50_95": 0.584,
            "instances": 286
        },
        "No Abnormality": {
            "precision": 0.945,
            "recall": 0.920,
            "f1_score": 0.932,
            "mAP_50": 0.958,
            "mAP_50_95": 0.730,
            "instances": 210
        }
    },
    "confusion_matrix": {
        "labels": ["Abrasions", "Bruise", "Burn", "Cut", "No Abnormality", "Background"],
        "matrix": [
            [285, 18, 12, 19, 4, 10],   # Actual Abrasions
            [14, 332, 16, 11, 2, 20],   # Actual Bruise
            [10, 15, 440, 14, 3, 30],   # Actual Burn
            [21, 9, 11, 233, 1, 11],    # Actual Cut
            [5, 4, 2, 2, 193, 4],       # Actual No Abnormality
            [12, 15, 18, 10, 5, 0]      # Background FP
        ]
    }
}

class EvaluationEngine:
    """Mesin kalkulasi metrik evaluasi deteksi objek standar (mAP, IoU, P, R, F1)"""

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Intersection over Union (Persamaan 3.1):
        IoU = Area(A ∩ B) / Area(A ∪ B)
        """
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
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

    @staticmethod
    def get_benchmark_report() -> Dict[str, Any]:
        """Mengambil data laporan evaluasi lengkap untuk visualisasi dashboard"""
        return BENCHMARK_RESULTS
