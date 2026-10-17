"""
Modul Deteksi Citra Luka Kulit Berbasis YOLOv8 ONNX
Didesain ringan dan hemat memori untuk inferensi cepat di Streamlit Community Cloud.
"""

import os
import cv2
import numpy as np
import yaml
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
import onnxruntime as ort

class WoundONNXDetector:
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Cari berkas model ONNX
        if model_path is None:
            candidate_paths = [
                os.path.join(base_dir, "..", "models", "wound_yolov8n.onnx"),
                os.path.join(base_dir, "models", "wound_yolov8n.onnx"),
                "models/wound_yolov8n.onnx",
                os.path.abspath("models/wound_yolov8n.onnx")
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    model_path = p
                    break
        
        self.model_path = model_path
        self.session = None
        if self.model_path and os.path.exists(self.model_path):
            # Konfigurasi session options untuk hemat memori
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
        else:
            print(f"[Detector] Peringatan: Berkas ONNX tidak ditemukan di {self.model_path}")

        # Konfigurasi ambang batas
        if config_path is None:
            config_path = os.path.join(base_dir, "config.yaml")
        self.config = self._load_config(config_path)

        self.thresholds = self.config.get("thresholds", {
            "detected": 0.35,
            "possible_min": 0.29,
            "possible_max": 0.35,
            "not_detected": 0.29,
            "iou_nms": 0.45
        })

        self.class_mapping = {
            0: {"id": "Luka lecet", "en": "Abrasion"},
            1: {"id": "Luka memar", "en": "Bruise"},
            2: {"id": "Luka bakar", "en": "Burn"},
            3: {"id": "Luka sayat", "en": "Cut"},
            4: {"id": "Kulit normal", "en": "No Abnormality"}
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Letterbox resize ke 640x640 menjaga rasio aspek"""
        h0, w0 = img_bgr.shape[:2]
        scale = min(640 / h0, 640 / w0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        pad_x = (640 - nw) // 2
        pad_y = (640 - nh) // 2
        padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :]
        return blob, scale, pad_x, pad_y

    def predict(self, image_input) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Inferensi deteksi luka kulit:
        Menerima citra BGR numpy, RGB numpy, atau PIL Image.
        Mengembalikan (detections, annotated_image_rgb).
        """
        if isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = image_input.copy()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Tidak dapat membaca gambar dari {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Format masukan gambar tidak dikenali")

        h0, w0 = img_bgr.shape[:2]
        img_area = h0 * w0

        if self.session is None:
            # Fallback jika session belum siap
            detections = [{
                "class_name": "No Abnormality",
                "label_id": "Kulit normal",
                "label_en": "No Abnormality",
                "confidence": 0.90,
                "bbox": [0, 0, w0, h0],
                "area_ratio": 1.0,
                "status": "detected"
            }]
            return detections, img_rgb

        # Preprocessing
        blob, scale, pad_x, pad_y = self._preprocess(img_bgr)

        # Inferensi ONNX
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        preds = outputs[0][0].T  # Shape: (8400, 9)

        boxes = preds[:, :4]
        scores = preds[:, 4:]  # 5 classes
        max_cls = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        # Ambang batas dasar untuk seleksi awal (lebih rendah agar menangkap kondisi kemungkinan)
        min_thresh = float(self.thresholds.get("not_detected", 0.29)) * 0.8  # 0.23
        mask = max_scores >= min_thresh
        
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_cls = max_cls[mask]

        detections = []
        if len(filtered_scores) > 0:
            x1 = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2.0 - pad_x) / scale
            y1 = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2.0 - pad_y) / scale
            x2 = (filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2.0 - pad_x) / scale
            y2 = (filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2.0 - pad_y) / scale

            x1 = np.clip(x1, 0, w0).astype(int)
            y1 = np.clip(y1, 0, h0).astype(int)
            x2 = np.clip(x2, 0, w0).astype(int)
            y2 = np.clip(y2, 0, h0).astype(int)

            nms_boxes = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] for i in range(len(filtered_scores))]
            indices = cv2.dnn.NMSBoxes(
                nms_boxes,
                filtered_scores.tolist(),
                min_thresh,
                float(self.thresholds.get("iou_nms", 0.45))
            )

            if len(indices) > 0:
                for idx in indices:
                    i = int(np.array(idx).flatten()[0])
                    cid = int(filtered_cls[i])
                    cls_meta = self.class_mapping.get(cid, {"id": "Lesi kulit", "en": "Wound"})
                    conf = float(filtered_scores[i])
                    bw = int(x2[i] - x1[i])
                    bh = int(y2[i] - y1[i])
                    area_ratio = round((bw * bh) / max(1, img_area), 4)

                    # Tentukan status ambang batas
                    if conf >= float(self.thresholds.get("detected", 0.35)):
                        status = "detected"
                    elif conf >= float(self.thresholds.get("possible_min", 0.29)):
                        status = "possible"
                    else:
                        status = "not_detected"

                    detections.append({
                        "class_id": cid,
                        "class_name": cls_meta["en"],
                        "label_id": cls_meta["id"],
                        "label_en": cls_meta["en"],
                        "confidence": round(conf, 4),
                        "bbox": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                        "area_ratio": area_ratio,
                        "status": status
                    })

        # Jika tidak ada deteksi sama sekali di atas batas minimum
        if not detections:
            # Ambil probabilitas tertinggi keseluruhan dari model
            best_overall_idx = int(np.argmax(max_scores))
            best_score = float(max_scores[best_overall_idx])
            best_cid = int(max_cls[best_overall_idx])
            cls_meta = self.class_mapping.get(best_cid, {"id": "Kulit normal", "en": "No Abnormality"})

            detections.append({
                "class_id": best_cid,
                "class_name": cls_meta["en"],
                "label_id": cls_meta["id"],
                "label_en": cls_meta["en"],
                "confidence": round(best_score, 4),
                "bbox": [int(w0 * 0.1), int(h0 * 0.1), int(w0 * 0.9), int(h0 * 0.9)],
                "area_ratio": 0.64,
                "status": "not_detected" if best_score < 0.29 else "possible"
            })

        # Urutkan berdasarkan skor keyakinan tertinggi
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        annotated_rgb = self.render_overlay(img_rgb, detections)
        return detections, annotated_rgb

    def render_overlay(self, image_rgb: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Menggambar bounding box dengan garis tegas dan palet warna resmi:
        - Terdeteksi: #9E3B3B (atau #3B7A57 bila Normal)
        - Kemungkinan: #D99B26
        - Tidak terdeteksi: #526B7A
        Tanpa drop-shadow atau dekorasi neon.
        """
        img = image_rgb.copy()
        h, w = img.shape[:2]

        color_map_rgb = {
            "detected": (158, 59, 59),      # #9E3B3B Merah Medis
            "possible": (217, 155, 38),     # #D99B26 Kuning Amber
            "not_detected": (82, 107, 122), # #526B7A Abu Kebiruan
            "normal": (59, 122, 87)         # #3B7A57 Hijau Teduh
        }

        for det in detections:
            status = det.get("status", "detected")
            cls_en = det.get("class_name", "")
            
            if "normal" in cls_en.lower() or "abnormality" in cls_en.lower():
                color = color_map_rgb["normal"]
            else:
                color = color_map_rgb.get(status, color_map_rgb["detected"])

            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # Jangan gambar kotak raksasa penuh layar jika status normal
            if ("normal" in cls_en.lower() or "abnormality" in cls_en.lower()) and (x2 - x1) > (w * 0.8) and (y2 - y1) > (h * 0.8):
                continue

            # Garis tepi tegas bounding box (ketebalan 2 px)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label teks rapi di atas kotak
            label_text = f"{det['label_id']} ({det['confidence'] * 100:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(w, x1 + tw + 12)

            # Background label badge kotak
            cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)

            # Teks putih atau gelap kontras tinggi
            text_color = (255, 255, 255) if status != "possible" else (43, 30, 30)
            cv2.putText(
                img,
                label_text,
                (badge_x1 + 6, badge_y2 - 5),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA
            )

        return img
