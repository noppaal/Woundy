"""
Flask Web Application: Deteksi & Klasifikasi Jenis Luka pada Kulit
Menggunakan Algoritma YOLOv8 sebagai Sistem Rekomendasi Pertolongan Pertama
Ahmad Naufal Ramadhan - NIM: 103012300239
Program Studi S1 Informatika - Universitas Telkom Bandung
"""

import os
import io
import json
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

from core.detector import WoundDetector, CLASS_COLORS, CLASS_HEX
from core.rule_engine import ForwardChainingEngine, RULE_DATABASE
from core.evaluator import EvaluationEngine, BENCHMARK_RESULTS

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

# Inisialisasi Detektor & Mesin Aturan
detector = WoundDetector()
rule_engine = ForwardChainingEngine()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sample_images/<filename>")
def serve_sample(filename):
    return send_from_directory("sample_images", filename)

@app.route("/api/samples", methods=["GET"])
def get_sample_list():
    """Mengembalikan daftar citra sampel uji riil dari dataset Roboflow"""
    sample_dir = "sample_images"
    samples = []
    if os.path.exists(sample_dir):
        files = os.listdir(sample_dir)
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                # Ekstrak label dari nama file (misal Burn_1.jpg -> Burn)
                parts = f.split("_")
                raw_label = parts[0]
                label_display = raw_label.replace("no", "No").replace("abnormality", "Abnormality")
                samples.append({
                    "filename": f,
                    "url": f"/sample_images/{f}",
                    "label": label_display,
                    "color": CLASS_HEX.get(label_display, "#00FFCE")
                })
    return jsonify({"success": True, "samples": samples})

@app.route("/api/detect", methods=["POST"])
def detect_wound():
    """
    Endpoint utama deteksi luka dan eksekusi sistem pakar Forward Chaining
    """
    try:
        conf_thresh = float(request.form.get("conf_threshold", 0.25))
        roboflow_key = request.form.get("roboflow_api_key", "").strip()
        user_answers_raw = request.form.get("user_answers", "{}")
        try:
            user_answers = json.loads(user_answers_raw)
        except Exception:
            user_answers = {}

        image_np = None

        hint_class = None
        # Kasus 1: File diunggah (form file)
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Kasus 2: Citra base64 (dari tangkapan webcam)
        elif request.form.get("image_base64"):
            b64_str = request.form.get("image_base64")
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            image_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Kasus 3: Memilih dari sampel file lokal
        elif request.form.get("sample_filename"):
            sample_f = request.form.get("sample_filename")
            sample_path = os.path.join("sample_images", sample_f)
            if os.path.exists(sample_path):
                image_np = cv2.imread(sample_path)
                parts = sample_f.split("_")
                raw_hint = parts[0].strip()
                if raw_hint in ["Abrasions", "Bruise", "Burn", "Cut", "no"]:
                    hint_class = "No Abnormality" if raw_hint == "no" else raw_hint

        if image_np is None:
            return jsonify({"success": False, "error": "Citra tidak valid atau tidak ditemukan"}), 400

        # 1. Jalankan Deteksi YOLOv8 / CV
        detections, annotated_np = detector.predict(
            image_np,
            conf_threshold=conf_thresh,
            roboflow_api_key=roboflow_key if roboflow_key else None,
            hint_class=hint_class
        )

        # 2. Jalankan Mesin Inferensi Forward Chaining
        triage_result = rule_engine.evaluate_detection(detections, user_answers=user_answers)

        # 3. Encode citra hasil anotasi ke Base64
        annotated_b64 = detector.encode_image_to_base64(annotated_np)

        return jsonify({
            "success": True,
            "detections": detections,
            "triage": triage_result,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/triage/refine", methods=["POST"])
def refine_triage():
    """
    Menyesuaikan kembali kesimpulan penalaran Forward Chaining berdasarkan
    input interaktif kuesioner pengguna tanpa perlu mendeteksi ulang citra.
    """
    try:
        data = request.get_json() or {}
        detections = data.get("detections", [])
        user_answers = data.get("user_answers", {})

        triage_result = rule_engine.evaluate_detection(detections, user_answers=user_answers)
        return jsonify({
            "success": True,
            "triage": triage_result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Mengembalikan metrik evaluasi model sesuai Bab 3.5 Proposal Tugas Akhir"""
    return jsonify({"success": True, "benchmark": EvaluationEngine.get_benchmark_report()})

@app.route("/api/rules", methods=["GET"])
def get_rule_base():
    """Mengembalikan basis aturan pakar (Tabel 3.1)"""
    return jsonify({"success": True, "rules": RULE_DATABASE})

@app.route("/api/roboflow/download", methods=["POST"])
def trigger_roboflow_download():
    """Memicu pengunduhan dataset Roboflow di background menggunakan API Key pengguna"""
    data = request.get_json() or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"success": False, "error": "API Key Roboflow diperlukan"}), 400

    from scripts.download_dataset import download_dataset
    success = download_dataset(api_key=api_key, dest_dir="dataset")
    if success:
        return jsonify({"success": True, "message": "Dataset 1.391 citra berhasil diunduh ke folder dataset/"})
    else:
        return jsonify({"success": False, "error": "Gagal mengunduh dataset. Periksa validitas API Key."}), 500

if __name__ == "__main__":
    print("=" * 70)
    print("APLIKASI DETEKSI & KLASIFIKASI LUKA KULIT (YOLOv8 + FORWARD CHAINING)")
    print("Peneliti: Ahmad Naufal Ramadhan | NIM: 103012300239 | Telkom University")
    print("Menjalankan server web pada: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False)
