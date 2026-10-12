"""
Skrip Pelatihan YOLOv8 untuk Deteksi & Klasifikasi Luka Kulit
Sesuai Metodologi Bab 3 Proposal Tugas Akhir
Ahmad Naufal Ramadhan - Universitas Telkom 2026
"""

import os
import sys
import argparse
from ultralytics import YOLO

def train_model(
    data_yaml: str = "dataset/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",  # GPU jika ada, atau 'cpu'
    project_dir: str = "runs/train",
    name: str = "wound_yolov8"
):
    print("=" * 60)
    print("Memulai Pelatihan YOLOv8 External Wound Detection")
    print(f"Data config : {data_yaml}")
    print(f"Base model  : {base_model}")
    print(f"Epochs      : {epochs}")
    print(f"Batch size  : {batch_size}")
    print(f"Image size  : {img_size}")
    print("=" * 60)

    import torch
    # Deteksi otomatis ketersediaan GPU CUDA
    if device not in ["cpu", None] and not torch.cuda.is_available():
        print(f"[INFO] CUDA GPU ('{device}') tidak tersedia pada instalasi PyTorch ini.")
        print("[INFO] Pelatihan dialihkan secara otomatis menggunakan CPU...")
        device = "cpu"
    else:
        print(f"Perangkat   : {'GPU ' + str(device) if device != 'cpu' else 'CPU'}")

    # Inisialisasi model YOLOv8
    model = YOLO(base_model)

    # Eksekusi pelatihan
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project_dir,
        name=name,
        patience=15,
        save=True,
        plots=True,
        verbose=True
    )

    # Simpan bobot terbaik ke direktori models/
    save_dir = str(getattr(results, "save_dir", ""))
    best_weights = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.join(project_dir, name, "weights", "best.pt")
    
    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        import shutil
        target_path = "models/wound_yolov8n.pt"
        shutil.copy(best_weights, target_path)
        print(f"\n[SUKSES] Bobot model terbaik disimpan ke: {target_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Wound Detection Model")
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    default_device = "0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device
    )
