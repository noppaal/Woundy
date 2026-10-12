"""
Skrip Evaluasi Model YOLOv8 pada Dataset Uji (Test Split: 137 citra)
Menghitung mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-Score
Ahmad Naufal Ramadhan - Universitas Telkom 2026
"""

import os
import sys
import argparse
from ultralytics import YOLO

def evaluate_model(
    model_path: str = "models/wound_yolov8n.pt",
    data_yaml: str = "dataset/data.yaml",
    split: str = "test",
    img_size: int = 640
):
    print("=" * 60)
    print("Evaluasi Model YOLOv8 External Wound Detection")
    print(f"Model path  : {model_path}")
    print(f"Data config : {data_yaml}")
    print(f"Split       : {split}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"[ERROR] Berkas model {model_path} tidak ditemukan!")
        return

    if not os.path.exists(data_yaml):
        print(f"[ERROR] Berkas {data_yaml} tidak ditemukan!")
        return

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=img_size,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("HASIL METRIK EVALUASI KUANTITATIF (BAB 3.5 PROPOSAL):")
    print("=" * 60)
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    
    p = metrics.box.mp
    r = metrics.box.mr
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
    print(f"F1-Score     : {f1:.4f}")
    print("=" * 60)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Wound Detection Model")
    parser.add_argument("--model", type=str, default="models/wound_yolov8n.pt")
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    evaluate_model(args.model, args.data, args.split, args.imgsz)
