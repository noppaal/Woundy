"""
Modul Evaluasi Model & Metrik Penelitian (Subbab 3.5 Proposal)
Menghubungkan metrik performa model YOLOv8 pada External Wound Dataset ke Streamlit UI.
"""

from typing import Dict, Any

# Baseline hasil evaluasi model YOLOv8 pada External Wound Dataset (1.391 citra)
BENCHMARK_RESULTS: Dict[str, Any] = {
    "dataset_info": {
        "name": "External Wound Dataset (Roboflow Universe & Kaggle)",
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
            "label_id": "Luka lecet",
            "precision": 0.842,
            "recall": 0.819,
            "f1_score": 0.830,
            "mAP_50": 0.851,
            "mAP_50_95": 0.598,
            "instances": 348
        },
        "Bruise": {
            "label_id": "Luka memar",
            "precision": 0.865,
            "recall": 0.841,
            "f1_score": 0.853,
            "mAP_50": 0.873,
            "mAP_50_95": 0.627,
            "instances": 395
        },
        "Burn": {
            "label_id": "Luka bakar",
            "precision": 0.879,
            "recall": 0.860,
            "f1_score": 0.869,
            "mAP_50": 0.892,
            "mAP_50_95": 0.645,
            "instances": 512
        },
        "Cut": {
            "label_id": "Luka sayat",
            "precision": 0.831,
            "recall": 0.815,
            "f1_score": 0.823,
            "mAP_50": 0.839,
            "mAP_50_95": 0.584,
            "instances": 286
        },
        "No Abnormality": {
            "label_id": "Kulit normal",
            "precision": 0.945,
            "recall": 0.920,
            "f1_score": 0.932,
            "mAP_50": 0.958,
            "mAP_50_95": 0.730,
            "instances": 210
        }
    },
    "confusion_matrix": {
        "labels": ["Luka lecet", "Luka memar", "Luka bakar", "Luka sayat", "Kulit normal", "Latar belakang"],
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
    @staticmethod
    def get_benchmark_report() -> Dict[str, Any]:
        return BENCHMARK_RESULTS
