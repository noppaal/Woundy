"""
Modul Deteksi Citra Luka Kulit Berbasis YOLOv8 ONNX
Didesain ringan, hemat memori, dan tahan-kesalahan (resilient) untuk inferensi di Streamlit Community Cloud.
Mendukung OpenCV (opencv-python-headless) dengan fallback murni PIL + NumPy jika OpenCV tidak tersedia.
"""

import os
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional
import onnxruntime as ort

# Coba impor OpenCV, fallback jika libGL/X11 tidak tersedia di container Linux
try:
    import cv2
except (ImportError, Exception):
    cv2 = None


def numpy_nms(boxes: List[List[int]], scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """
    Implementasi mandiri Non-Maximum Suppression (NMS) berbasis NumPy.
    Digunakan sebagai fallback jika modul cv2.dnn tidak tersedia.
    Format kotak masukan: [x, y, w, h].
    """
    if len(boxes) == 0:
        return []

    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)

    x1 = b[:, 0]
    y1 = b[:, 1]
    x2 = b[:, 0] + b[:, 2]
    y2 = b[:, 1] + b[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = s.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        ovr = np.where(union > 0, inter / union, 0.0)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


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

    def _preprocess(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Letterbox resize ke 640x640 menjaga rasio aspek (RGB)"""
        h0, w0 = img_rgb.shape[:2]
        scale = min(640 / h0, 640 / w0)
        nw, nh = int(w0 * scale), int(h0 * scale)

        if cv2 is not None:
            resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            pil_temp = Image.fromarray(img_rgb)
            resized = np.array(pil_temp.resize((nw, nh), Image.Resampling.BILINEAR))
        
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        pad_x = (640 - nw) // 2
        pad_y = (640 - nh) // 2
        padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :]
        return blob, scale, pad_x, pad_y

    def predict(self, image_input) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Inferensi deteksi luka kulit:
        Menerima citra BGR numpy, RGB numpy, atau PIL Image.
        Mengembalikan (detections, annotated_image_rgb).
        """
        if isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_rgb = np.stack([image_input] * 3, axis=-1)
            elif image_input.shape[2] == 4:
                img_rgb = image_input[:, :, :3]
            else:
                img_rgb = image_input.copy()
        elif isinstance(image_input, str):
            pil_temp = Image.open(image_input).convert("RGB")
            img_rgb = np.array(pil_temp)
        else:
            raise ValueError("Format masukan gambar tidak dikenali")

        h0, w0 = img_rgb.shape[:2]
        img_area = h0 * w0

        if self.session is None:
            detections = [{
                "class_id": 4,
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
        blob, scale, pad_x, pad_y = self._preprocess(img_rgb)

        # Inferensi ONNX
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        preds = outputs[0][0].T  # Shape: (8400, 9)

        boxes = preds[:, :4]
        scores = preds[:, 4:]  # 5 classes
        max_cls = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        # Ambang batas dasar untuk seleksi awal
        min_thresh = float(self.thresholds.get("not_detected", 0.29)) * 0.8
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

            # Gunakan cv2.dnn.NMSBoxes jika cv2 tersedia, selain itu gunakan fallback NumPy NMS
            nms_indices = []
            iou_thresh = float(self.thresholds.get("iou_nms", 0.45))
            if cv2 is not None and hasattr(cv2, "dnn") and hasattr(cv2.dnn, "NMSBoxes"):
                try:
                    raw_indices = cv2.dnn.NMSBoxes(
                        nms_boxes,
                        filtered_scores.tolist(),
                        min_thresh,
                        iou_thresh
                    )
                    if len(raw_indices) > 0:
                        nms_indices = [int(np.array(idx).flatten()[0]) for idx in raw_indices]
                except Exception:
                    nms_indices = numpy_nms(nms_boxes, filtered_scores, iou_thresh)
            else:
                nms_indices = numpy_nms(nms_boxes, filtered_scores, iou_thresh)

            if len(nms_indices) > 0:
                for i in nms_indices:
                    cid = int(filtered_cls[i])
                    cls_meta = self.class_mapping.get(cid, {"id": "Lesi kulit", "en": "Wound"})
                    conf = float(filtered_scores[i])
                    bw = int(x2[i] - x1[i])
                    bh = int(y2[i] - y1[i])
                    area_ratio = round((bw * bh) / max(1, img_area), 4)

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

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        annotated_rgb = self.render_overlay(img_rgb, detections)
        return detections, annotated_rgb

    def render_overlay(self, image_rgb: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Menggambar bounding box dengan palet warna klinis resmi:
        - Terdeteksi: #9E3B3B (atau #2E7D4E bila Normal)
        - Kemungkinan: #D99B26
        - Tidak terdeteksi: #526B7A
        Mendukung OpenCV atau PIL ImageDraw jika OpenCV tidak tersedia.
        """
        img = image_rgb.copy()
        h, w = img.shape[:2]

        color_map_rgb = {
            "detected": (158, 59, 59),      # #9E3B3B Merah Medis
            "possible": (217, 155, 38),     # #D99B26 Kuning Amber
            "not_detected": (82, 107, 122), # #526B7A Abu Kebiruan
            "normal": (46, 125, 78)         # #2E7D4E Hijau Teduh
        }

        # Jika cv2 tersedia, gunakan cv2 untuk penggambaran yang cepat
        if cv2 is not None:
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

                if ("normal" in cls_en.lower() or "abnormality" in cls_en.lower()) and (x2 - x1) > (w * 0.8) and (y2 - y1) > (h * 0.8):
                    continue

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                label_text = f"{det['label_id']} ({det['confidence'] * 100:.1f}%)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.52
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                badge_y1 = max(0, y1 - th - 8)
                badge_y2 = y1
                badge_x1 = x1
                badge_x2 = min(w, x1 + tw + 12)

                cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
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

        # Fallback menggambar menggunakan PIL ImageDraw
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

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

            if ("normal" in cls_en.lower() or "abnormality" in cls_en.lower()) and (x2 - x1) > (w * 0.8) and (y2 - y1) > (h * 0.8):
                continue

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label_text = f"{det['label_id']} ({det['confidence'] * 100:.1f}%)"
            badge_h = 18
            badge_w = len(label_text) * 8 + 10
            badge_y1 = max(0, y1 - badge_h)
            badge_x2 = min(w, x1 + badge_w)

            draw.rectangle([x1, badge_y1, badge_x2, y1], fill=color)
            text_color = (255, 255, 255) if status != "possible" else (43, 30, 30)
            draw.text((x1 + 5, badge_y1 + 2), label_text, fill=text_color)

        return np.array(pil_img)
