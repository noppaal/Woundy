"""
Pembersihan Kebocoran Data (Data Leakage) antar split dataset.

Mendeteksi citra pada valid/test yang identik atau sangat mirip (perceptual dHash)
dengan citra pada split lain, lalu MEMINDAHKANNYA (bukan menghapus) ke
dataset_removed/<split>/{images,labels} sehingga dapat dikembalikan.

Aturan:
  - Train tidak pernah diubah.
  - Citra valid yang mirip train      -> dipindahkan.
  - Citra test  yang mirip train/valid -> dipindahkan.

Pemakaian:
  python scripts/clean_leakage.py            # dry-run (hanya melaporkan)
  python scripts/clean_leakage.py --apply    # memindahkan berkas
  python scripts/clean_leakage.py --restore  # mengembalikan semua berkas
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE, "dataset")
REMOVED = os.path.join(BASE, "dataset_removed")


def dhash(path: str) -> np.ndarray:
    im = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def load_split(split: str):
    d = os.path.join(DATASET, split, "images")
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    hashes = np.array([dhash(os.path.join(d, f)) for f in files]) if files else np.zeros((0, 64), bool)
    return files, hashes


def find_leaks(src, ref_list, threshold):
    files, hashes = src
    leaks = []
    for i, f in enumerate(files):
        for ref_name, (_, ref_h) in ref_list:
            if len(ref_h) and ((ref_h != hashes[i]).sum(1) <= threshold).any():
                leaks.append((f, ref_name))
                break
    return leaks


def move(split, fname):
    stem = os.path.splitext(fname)[0]
    for sub, name in (("images", fname), ("labels", stem + ".txt")):
        src = os.path.join(DATASET, split, sub, name)
        if os.path.exists(src):
            dst_dir = os.path.join(REMOVED, split, sub)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, name))


def restore():
    n = 0
    for split in ("valid", "test"):
        for sub in ("images", "labels"):
            d = os.path.join(REMOVED, split, sub)
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                shutil.move(os.path.join(d, name), os.path.join(DATASET, split, sub, name))
                n += 1
    print(f"[INFO] {n} berkas dikembalikan.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--threshold", type=int, default=4, help="Jarak Hamming maksimum (64-bit dHash)")
    args = ap.parse_args()

    if args.restore:
        restore()
        return

    train, valid, test = load_split("train"), load_split("valid"), load_split("test")
    leaks = {
        "valid": find_leaks(valid, [("train", train)], args.threshold),
        "test": find_leaks(test, [("train", train), ("valid", valid)], args.threshold),
    }

    print(f"Train: {len(train[0])} | Valid: {len(valid[0])} | Test: {len(test[0])}")
    for split, items in leaks.items():
        print(f"[{split}] kebocoran terdeteksi: {len(items)}")
        for f, ref in items:
            print(f"    {f}  (mirip dengan {ref})")

    report = {s: [{"file": f, "similar_to": r} for f, r in v] for s, v in leaks.items()}
    if not args.apply:
        print("\n[DRY-RUN] Tidak ada berkas dipindahkan. Gunakan --apply untuk memindahkan.")
        return

    for split, items in leaks.items():
        for f, _ in items:
            move(split, f)
        cache = os.path.join(DATASET, split, "labels.cache")
        if os.path.exists(cache):
            os.remove(cache)  # cache label harus dibangun ulang oleh Ultralytics

    os.makedirs(REMOVED, exist_ok=True)
    with open(os.path.join(REMOVED, "removed_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[OK] Berkas dipindahkan ke {REMOVED}. Laporan: removed_report.json")


if __name__ == "__main__":
    main()
