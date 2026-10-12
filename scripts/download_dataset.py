"""
Skrip Pengunduhan Dataset dari Roboflow Universe
Mengunduh proyek: w-afwxp / wound-no-blister-2 (Versi 2)
Format: YOLOv8 (train 979, valid 275, test 137 citra)
Ahmad Naufal Ramadhan - Universitas Telkom 2026
"""

import os
import sys
import argparse

def download_dataset(api_key: str, dest_dir: str = "dataset"):
    print("=" * 60)
    print("Memulai proses download dataset dari Roboflow Universe...")
    print(f"Workspace: w-afwxp")
    print(f"Project: wound-no-blister-2")
    print(f"Version: 2 (YOLOv8 format)")
    print(f"Target Direktori: {dest_dir}")
    print("=" * 60)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Error: Library roboflow belum terpasang. Jalankan: pip install roboflow")
        return False

    os.makedirs(dest_dir, exist_ok=True)
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("w-afwxp").project("wound-no-blister-2")
        version = project.version(2)
        dataset = version.download(model_format="yolov8", location=dest_dir, overwrite=True)
        print("\n[SUKSES] Dataset berhasil diunduh ke:", dataset.location)
        print(f"File konfigurasi YOLOv8: {os.path.join(dest_dir, 'data.yaml')}")
        return True
    except Exception as e:
        print(f"\n[GAGAL] Terjadi kesalahan saat mengunduh dataset: {e}")
        print("Pastikan Roboflow API Key valid dan memiliki akses internet.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Roboflow wound-no-blister-2 dataset")
    parser.add_argument("--api-key", type=str, required=False, default=None, help="Roboflow Private/Publishable API Key")
    parser.add_argument("--dest", type=str, default="dataset", help="Direktori penyimpanan dataset")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        try:
            print("Roboflow API Key belum diberikan.")
            print("(Dapatkan gratis di https://app.roboflow.com/settings/api)")
            api_key = input("Masukkan Roboflow API Key Anda: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nDibatalkan.")
            sys.exit(1)

    # Bersihkan karakter pembungkus seperti <, >, tanda petik jika ada
    api_key = api_key.strip("<>\"' \t\r\n")

    if not api_key:
        print("[ERROR] API Key tidak boleh kosong.")
        sys.exit(1)

    success = download_dataset(api_key, args.dest)
    sys.exit(0 if success else 1)
